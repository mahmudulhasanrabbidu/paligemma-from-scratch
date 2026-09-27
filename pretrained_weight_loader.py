import os
import glob
import json
import gc
from typing import Dict, Tuple, Optional, Set
import torch
from safetensors import safe_open
from tqdm import tqdm

from gemma_new.config import GemmaConfig, PaliGemmaConfig
from gemma_new.model import GemmaForCausalLM, PaliGemmaForConditionalGeneration

class PretrainedGemmaModel(GemmaForCausalLM):
    @classmethod
    def create_custom_config(cls, config_path: str = "config.json", dtype: torch.dtype = torch.bfloat16) -> GemmaConfig:
        print(f"Reading configuration from: {config_path}")
        return GemmaConfig.from_json_file(config_path, dtype=dtype)

    @staticmethod
    def get_safetensors_files(model_path: str) -> list:
        """Discovers safetensors files from a path or directory."""
        if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
            return [model_path]
        if os.path.isdir(model_path):
            files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
            if not files:
                raise FileNotFoundError(f"No *.safetensors files found in: '{model_path}'")
            return files
        raise FileNotFoundError(f"Invalid model_path: '{model_path}'")

    def visualize_custom_parameters(self) -> None:
        print("\n--- Custom Model Parameter Layout ---")
        total_params = 0
        for name, param in self.named_parameters():
            print(f"  Layer: {name:<55} | Shape: {str(tuple(param.shape)):<20} | Dtype: {param.dtype}")
            total_params += param.numel()
        print(f"\nTotal Architecture Parameters: {total_params:,}")
        print("--------------------------------------\n")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple["PretrainedGemmaModel", Set[str]]:
        """
        Master initialization method following Reasoning-LLM style:
        1. Builds typed GemmaConfig dataclass.
        2. Instantiates empty PyTorch model architecture in bfloat16 to conserve RAM.
        3. Identifies parameter mapping table between Hugging Face keys and model parameters.
        4. Ingests safetensors shards sequentially, validating shapes and copying in-place.
        5. Ties weights and verifies parameter coverage.
        """
        # Resolve configuration path
        if config_path is None:
            if os.path.isdir(model_path):
                config_path = os.path.join(model_path, "config.json")
            else:
                config_path = os.path.join(os.path.dirname(model_path), "config.json")

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"config.json not found at '{config_path}'")

        # Build Configuration
        print("Building configuration from config.json...")
        config = cls.create_custom_config(config_path, dtype=dtype)
        config.dtype = dtype

        # Instantiate Architecture in bfloat16 to avoid RAM exhaustion
        print(f"Initializing empty Gemma architecture on device: {device} (dtype={config.dtype})...")
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.dtype)
        model = cls(config).to(device)
        torch.set_default_dtype(old_dtype)

        # Discover Safetensors files
        safetensors_files = cls.get_safetensors_files(model_path)
        print(f"Found {len(safetensors_files)} safetensors file(s) to process.")

        # Inspect sample keys to determine prefix
        with safe_open(safetensors_files[0], framework="pt", device="cpu") as sample_f:
            sample_keys = list(sample_f.keys())

        if any(k.startswith("language_model.model.") for k in sample_keys):
            prefix = "language_model.model."
            lm_head_key = "language_model.lm_head.weight"
            print("Detected PaliGemma checkpoint format (using prefix: 'language_model.model.')")
        elif any(k.startswith("model.") for k in sample_keys):
            prefix = "model."
            lm_head_key = "lm_head.weight"
            print("Detected Standalone Gemma checkpoint format (using prefix: 'model.')")
        else:
            prefix = ""
            lm_head_key = "lm_head.weight"

        # Build Target Mapping Table (Hugging Face Key -> Model Parameter)
        target_map: Dict[str, torch.nn.Parameter] = {
            f"{prefix}embed_tokens.weight": model.model.embed_tokens.weight,
            f"{prefix}norm.weight": model.model.norm.weight,
        }
        if lm_head_key:
            target_map[lm_head_key] = model.lm_head.weight

        for i in range(config.num_hidden_layers):
            block = model.model.layers[i]
            p = f"{prefix}layers.{i}"
            target_map[f"{p}.input_layernorm.weight"] = block.input_layernorm.weight
            target_map[f"{p}.post_attention_layernorm.weight"] = block.post_attention_layernorm.weight
            target_map[f"{p}.self_attn.q_proj.weight"] = block.self_attn.q_proj.weight
            target_map[f"{p}.self_attn.k_proj.weight"] = block.self_attn.k_proj.weight
            target_map[f"{p}.self_attn.v_proj.weight"] = block.self_attn.v_proj.weight
            target_map[f"{p}.self_attn.o_proj.weight"] = block.self_attn.o_proj.weight
            target_map[f"{p}.mlp.gate_proj.weight"] = block.mlp.gate_proj.weight
            target_map[f"{p}.mlp.up_proj.weight"] = block.mlp.up_proj.weight
            target_map[f"{p}.mlp.down_proj.weight"] = block.mlp.down_proj.weight

        # Ingest Shards & Inject Weights In-Place (Reasoning-LLM Style)
        print("Mapping and injecting weights in-place...")
        assigned_keys: Set[str] = set()
        total_injected_elements = 0

        for sf in tqdm(safetensors_files, desc="Processing Weight Shards"):
            with safe_open(sf, framework="pt", device="cpu") as f:
                for hf_key in f.keys():
                    if hf_key in target_map:
                        source_tensor = f.get_tensor(hf_key)
                        target_param = target_map[hf_key]

                        # Shape validation
                        if target_param.shape != source_tensor.shape:
                            raise ValueError(
                                f"Shape mismatch at '{hf_key}'! Expected {target_param.shape}, got {source_tensor.shape}"
                            )

                        # In-place tensor copy without autograd graph building or duplicate allocation
                        with torch.no_grad():
                            target_param.copy_(source_tensor)

                        assigned_keys.add(hf_key)
                        total_injected_elements += source_tensor.numel()
                        del source_tensor

            # Clean memory after each shard
            gc.collect()

        # Tie weights if lm_head wasn't explicitly in checkpoint
        model.tie_weights()

        print("Pretrained weights loaded and assigned successfully!\n")

        # Verification Checks
        num_model_params = sum(p.numel() for p in model.parameters())

        print("--- Parameter Coverage Verification ---")
        print(f"Total Model Parameters:          {num_model_params:,}")
        print(f"Total Injected Weight Elements:  {total_injected_elements:,}")
        print(f"Target Keys Assigned:            {len(assigned_keys)} / {len(target_map)}")

        if len(assigned_keys) == len(target_map) or (len(assigned_keys) == len(target_map) - 1 and lm_head_key not in assigned_keys):
            print("Status: PERFECT MATCH! All Gemma layers successfully injected.")
        else:
            missing = set(target_map.keys()) - assigned_keys
            print(f"Status: Incomplete mapping. Missing keys: {missing}")

        return model, assigned_keys


class PretrainedPaliGemmaModel(PaliGemmaForConditionalGeneration):
    @classmethod
    def create_custom_config(
        cls, config_path: str = "config.json", dtype: torch.dtype = torch.bfloat16
    ) -> PaliGemmaConfig:
        config = PaliGemmaConfig.from_json_file(config_path)
        config.text_config.dtype = dtype
        config.vision_config.dtype = dtype
        return config

    def visualize_custom_parameters(self) -> None:
        """Prints the names, shapes, and total count of custom model parameters."""
        print("\n--- PaliGemma Custom Parameter Layout ---")
        total_params = 0
        for name, param in self.named_parameters():
            print(f"  Layer: {name:<65} | Shape: {str(tuple(param.shape)):<20} | Dtype: {param.dtype}")
            total_params += param.numel()
        print(f"\nTotal Architecture Parameters: {total_params:,}")
        print("-------------------------------------------\n")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple["PretrainedPaliGemmaModel", Set[str]]:
        # Resolve configuration path
        if config_path is None:
            if os.path.isdir(model_path):
                config_path = os.path.join(model_path, "config.json")
            else:
                config_path = os.path.join(os.path.dirname(model_path), "config.json")

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"config.json not found at '{config_path}'")

        # Build typed PaliGemmaConfig
        print(f"Reading PaliGemma configuration from: {config_path}")
        config = cls.create_custom_config(config_path, dtype=dtype)

        # Instantiate Architecture in requested dtype
        print(f"Initializing empty PaliGemma architecture on device: {device} (dtype={dtype})...")
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        model = cls(config).to(device)
        torch.set_default_dtype(old_dtype)

        # Discover Safetensors files
        safetensors_files = PretrainedGemmaModel.get_safetensors_files(model_path)
        print(f"Found {len(safetensors_files)} safetensors file(s) to process.")

        # Build Target Mapping Table (Hugging Face Key -> Model Parameter)
        target_map: Dict[str, torch.nn.Parameter] = {}

        # Vision Tower Mapping
        v_model = model.vision_tower.vision_model
        target_map["vision_tower.vision_model.embeddings.patch_embedding.weight"] = v_model.embeddings.patch_embedding.weight
        target_map["vision_tower.vision_model.embeddings.patch_embedding.bias"] = v_model.embeddings.patch_embedding.bias
        target_map["vision_tower.vision_model.embeddings.position_embedding.weight"] = v_model.embeddings.position_embedding.weight

        for i in range(config.vision_config.num_hidden_layers):
            layer = v_model.encoder.layers[i]
            p = f"vision_tower.vision_model.encoder.layers.{i}"
            target_map[f"{p}.layer_norm1.weight"] = layer.layer_norm1.weight
            target_map[f"{p}.layer_norm1.bias"] = layer.layer_norm1.bias
            target_map[f"{p}.self_attn.q_proj.weight"] = layer.self_attn.q_proj.weight
            target_map[f"{p}.self_attn.q_proj.bias"] = layer.self_attn.q_proj.bias
            target_map[f"{p}.self_attn.k_proj.weight"] = layer.self_attn.k_proj.weight
            target_map[f"{p}.self_attn.k_proj.bias"] = layer.self_attn.k_proj.bias
            target_map[f"{p}.self_attn.v_proj.weight"] = layer.self_attn.v_proj.weight
            target_map[f"{p}.self_attn.v_proj.bias"] = layer.self_attn.v_proj.bias
            target_map[f"{p}.self_attn.out_proj.weight"] = layer.self_attn.out_proj.weight
            target_map[f"{p}.self_attn.out_proj.bias"] = layer.self_attn.out_proj.bias
            target_map[f"{p}.layer_norm2.weight"] = layer.layer_norm2.weight
            target_map[f"{p}.layer_norm2.bias"] = layer.layer_norm2.bias
            target_map[f"{p}.mlp.fc1.weight"] = layer.mlp.fc1.weight
            target_map[f"{p}.mlp.fc1.bias"] = layer.mlp.fc1.bias
            target_map[f"{p}.mlp.fc2.weight"] = layer.mlp.fc2.weight
            target_map[f"{p}.mlp.fc2.bias"] = layer.mlp.fc2.bias

        target_map["vision_tower.vision_model.post_layernorm.weight"] = v_model.post_layernorm.weight
        target_map["vision_tower.vision_model.post_layernorm.bias"] = v_model.post_layernorm.bias

        # MultiModal Projector Mapping
        target_map["multi_modal_projector.linear.weight"] = model.multi_modal_projector.linear.weight
        target_map["multi_modal_projector.linear.bias"] = model.multi_modal_projector.linear.bias

        # Language Model Mapping
        lm_model = model.language_model.model
        target_map["language_model.model.embed_tokens.weight"] = lm_model.embed_tokens.weight
        target_map["language_model.model.norm.weight"] = lm_model.norm.weight

        for i in range(config.text_config.num_hidden_layers):
            block = lm_model.layers[i]
            p = f"language_model.model.layers.{i}"
            target_map[f"{p}.input_layernorm.weight"] = block.input_layernorm.weight
            target_map[f"{p}.post_attention_layernorm.weight"] = block.post_attention_layernorm.weight
            target_map[f"{p}.self_attn.q_proj.weight"] = block.self_attn.q_proj.weight
            target_map[f"{p}.self_attn.k_proj.weight"] = block.self_attn.k_proj.weight
            target_map[f"{p}.self_attn.v_proj.weight"] = block.self_attn.v_proj.weight
            target_map[f"{p}.self_attn.o_proj.weight"] = block.self_attn.o_proj.weight
            target_map[f"{p}.mlp.gate_proj.weight"] = block.mlp.gate_proj.weight
            target_map[f"{p}.mlp.up_proj.weight"] = block.mlp.up_proj.weight
            target_map[f"{p}.mlp.down_proj.weight"] = block.mlp.down_proj.weight

        # Ingest Shards & Inject Weights In-Place
        print(f"Mapping and injecting {len(target_map)} weights in-place...")
        assigned_keys: Set[str] = set()
        total_injected_elements = 0

        for sf in tqdm(safetensors_files, desc="Processing PaliGemma Shards"):
            with safe_open(sf, framework="pt", device="cpu") as f:
                for hf_key in f.keys():
                    if hf_key in target_map:
                        source_tensor = f.get_tensor(hf_key)
                        target_param = target_map[hf_key]

                        if target_param.shape != source_tensor.shape:
                            raise ValueError(
                                f"Shape mismatch at '{hf_key}'! Expected {target_param.shape}, got {source_tensor.shape}"
                            )

                        with torch.no_grad():
                            target_param.copy_(source_tensor)

                        assigned_keys.add(hf_key)
                        total_injected_elements += source_tensor.numel()
                        del source_tensor

            gc.collect()

        # Tie language model weights
        model.tie_weights()

        print("PaliGemma pretrained weights loaded and assigned successfully!\n")

        # Verification Checks
        num_model_params = sum(p.numel() for p in model.parameters())

        print("--- Parameter Coverage Verification ---")
        print(f"Total Model Parameters:          {num_model_params:,}")
        print(f"Total Injected Weight Elements:  {total_injected_elements:,}")
        print(f"Target Keys Assigned:            {len(assigned_keys)} / {len(target_map)}")

        if len(assigned_keys) == len(target_map):
            print("Status: PERFECT MATCH! All PaliGemma components (vision + projector + language) successfully injected.")
        else:
            missing = set(target_map.keys()) - assigned_keys
            print(f"Status: Incomplete mapping. Missing keys: {missing}")

        return model, assigned_keys
