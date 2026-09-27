from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json
import torch

@dataclass
class GemmaConfig:
    """
    Configuration dataclass for the Gemma causal language model.
    Corresponds to Gemma / Gemma 2 / PaliGemma language backbones.
    """
    vocab_size: int = 257216
    hidden_size: int = 2048
    intermediate_size: int = 16384
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: Optional[int] = 0
    dtype: torch.dtype = torch.bfloat16
    num_image_tokens: Optional[int] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], dtype: Optional[torch.dtype] = None) -> "GemmaConfig":
        """Instantiates GemmaConfig from a Python dictionary."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype is not None:
            torch_dtype = dtype
        else:
            raw_dtype = config_dict.get("torch_dtype", "bfloat16")
            torch_dtype = dtype_map.get(raw_dtype, torch.bfloat16)

        return cls(
            vocab_size=config_dict.get("vocab_size", 257216),
            hidden_size=config_dict.get("hidden_size", 2048),
            intermediate_size=config_dict.get("intermediate_size", 16384),
            num_hidden_layers=config_dict.get("num_hidden_layers", 18),
            num_attention_heads=config_dict.get("num_attention_heads", 8),
            num_key_value_heads=config_dict.get("num_key_value_heads", 1),
            head_dim=config_dict.get("head_dim", 256),
            max_position_embeddings=config_dict.get("max_position_embeddings", 8192),
            rms_norm_eps=config_dict.get("rms_norm_eps", 1e-6),
            rope_theta=config_dict.get("rope_theta", 10000.0),
            attention_bias=config_dict.get("attention_bias", False),
            attention_dropout=config_dict.get("attention_dropout", 0.0),
            pad_token_id=config_dict.get("pad_token_id", 0),
            dtype=torch_dtype,
            num_image_tokens=config_dict.get("num_image_tokens", None),
        )

    @classmethod
    def from_json_file(cls, json_path: str, dtype: Optional[torch.dtype] = torch.bfloat16) -> "GemmaConfig":
        """Reads a Hugging Face config.json file and parses GemmaConfig."""
        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # Handle PaliGemma config having a nested 'text_config'
        if "text_config" in cfg and isinstance(cfg["text_config"], dict):
            cfg = cfg["text_config"]
        return cls.from_dict(cfg, dtype=dtype)


@dataclass
class SiglipVisionConfig:
    """
    Configuration dataclass for the SigLIP Vision Transformer backbone.
    """
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 14
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    num_image_tokens: int = 256
    projection_dim: int = 2048
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SiglipVisionConfig":
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        raw_dtype = config_dict.get("torch_dtype", "bfloat16")
        torch_dtype = dtype_map.get(raw_dtype, torch.bfloat16)

        return cls(
            hidden_size=config_dict.get("hidden_size", 1152),
            intermediate_size=config_dict.get("intermediate_size", 4304),
            num_hidden_layers=config_dict.get("num_hidden_layers", 27),
            num_attention_heads=config_dict.get("num_attention_heads", 16),
            num_channels=config_dict.get("num_channels", 3),
            image_size=config_dict.get("image_size", 224),
            patch_size=config_dict.get("patch_size", 14),
            layer_norm_eps=config_dict.get("layer_norm_eps", 1e-6),
            attention_dropout=config_dict.get("attention_dropout", 0.0),
            num_image_tokens=config_dict.get("num_image_tokens", 256),
            projection_dim=config_dict.get("projection_dim", 2048),
            dtype=torch_dtype,
        )


@dataclass
class PaliGemmaConfig:
    """
    Top-level compositional configuration binding SigLIP and Gemma.
    """
    text_config: GemmaConfig = field(default_factory=GemmaConfig)
    vision_config: SiglipVisionConfig = field(default_factory=SiglipVisionConfig)
    ignore_index: int = -100
    image_token_index: int = 257152
    vocab_size: int = 257216
    projection_dim: int = 2048
    hidden_size: int = 2048
    pad_token_id: Optional[int] = 0

    @classmethod
    def from_json_file(cls, json_path: str) -> "PaliGemmaConfig":
        with open(json_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

        text_dict = raw_cfg.get("text_config", {})
        vision_dict = raw_cfg.get("vision_config", {})

        text_cfg = GemmaConfig.from_dict(text_dict)
        vision_cfg = SiglipVisionConfig.from_dict(vision_dict)

        return cls(
            text_config=text_cfg,
            vision_config=vision_cfg,
            ignore_index=raw_cfg.get("ignore_index", -100),
            image_token_index=raw_cfg.get("image_token_index", 257152),
            vocab_size=raw_cfg.get("vocab_size", text_cfg.vocab_size),
            projection_dim=raw_cfg.get("projection_dim", 2048),
            hidden_size=raw_cfg.get("hidden_size", 2048),
            pad_token_id=raw_cfg.get("pad_token_id", 0),
        )
