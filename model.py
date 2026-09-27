import math
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
from gemma_new.config import GemmaConfig, PaliGemmaConfig
from gemma_new.siglip import SiglipVisionModel


class KVCache:
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        # key_cache[0]: [B, num_kv_heads, seq_len, head_dim] -> seq_len
        return self.key_cache[0].shape[-2]

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # key_states, value_states: [B, num_kv_heads, S, head_dim]
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # [B, num_kv_heads, S_past, head_dim] cat [B, num_kv_heads, S_new, head_dim] -> [B, num_kv_heads, S_past + S_new, head_dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reset(self) -> None:
        self.key_cache.clear()
        self.value_cache.clear()


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # weight: [dim]
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, dim]
        # x.pow(2).mean(-1, keepdim=True): [B, S, 1]
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, dim]
        output = self._norm(x.float())
        # [B, S, dim] * [dim] -> [B, S, dim]
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        theta_i = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)) # [d/2]
        self.register_buffer("theta_i", theta_i, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [B, num_attn_heads, seq_len, head_dim]
        # position_ids: [B, seq_len]
        self.theta_i.to(x.device)
        
        # [d/2] -> [1, d/2, 1] -> [B, d/2, 1]
        theta_expanded = self.theta_i[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # [B, seq_len] -> [B, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            # [B, d/2, 1] @ [B, 1, seq_len] = [B, d/2, seq_len] -> [B, seq_len, d/2]
            m_tht = (theta_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            
            # [B, seq_len, d]
            embd = torch.cat([m_tht, m_tht], dim=-1)

            cos = embd.cos()
            sin = embd.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    # x: [B, num_attn_head, S, head_dim]
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    # [B, num_attn_head, S, head_dim]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim = 1):
    # cos, sin: [B, S, head_dim]
    # [B, S, head_dim] -> [B, 1, S, head_dim]
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    # q, k: [B, num_heads, S, head_dim]
    # [B, 1, S, head_dim] * [B, num_heads, S, head_dim] -> [B, num_heads, S, head_dim]
    q_embd = q * cos + rotate_half(q) * sin
    k_embd = k * cos + rotate_half(k) * sin

    return q_embd, k_embd


class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=config.dtype)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=config.dtype)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=config.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, hidden_size]
        # gate_proj(x): [B, S, intermediate_size]
        # up_proj(x): [B, S, intermediate_size]
        # [B, S, intermediate_size] * [B, S, intermediate_size] -> [B, S, intermediate_size]
        gate_up = nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        # down_proj(gate_up): [B, S, hidden_size]
        return self.down_proj(gate_up)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # hidden_states: [B, num_kv_heads, S, head_dim]
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # [B, num_kv_heads, S, head_dim] -> [B, num_kv_heads, 1, S, head_dim] -> [B, num_kv_heads, n_rep, S, head_dim]
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # [B, num_kv_heads * n_rep, S, head_dim]
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias, dtype=config.dtype)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, dtype=config.dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, dtype=config.dtype)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias, dtype=config.dtype)

        self.rotary_emb = GemmaRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: [B, Q_len, hidden_size]
        bsz, q_len, _ = hidden_states.size()

        # [B, Q_len, num_heads * head_dim]
        query_states = self.q_proj(hidden_states)
        # [B, Q_len, num_kv_heads * head_dim]
        key_states = self.k_proj(hidden_states)
        # [B, Q_len, num_kv_heads * head_dim]
        value_states = self.v_proj(hidden_states)

        # [B, Q_len, num_heads * head_dim] -> [B, Q_len, num_heads, head_dim] -> [B, num_heads, Q_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, Q_len, num_kv_heads * head_dim] -> [B, Q_len, num_kv_heads, head_dim] -> [B, num_kv_heads, Q_len, head_dim]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # [B, Q_len, num_kv_heads * head_dim] -> [B, Q_len, num_kv_heads, head_dim] -> [B, num_kv_heads, Q_len, head_dim]
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_ids is None:
            # [1, Q_len] -> [B, Q_len]
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        # cos, sin: [B, Q_len, head_dim]
        cos, sin = self.rotary_emb(value_states, position_ids)
        # query_states: [B, num_heads, Q_len, head_dim], key_states: [B, num_kv_heads, Q_len, head_dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            # key_states, value_states: [B, num_kv_heads, KV_len, head_dim]
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # [B, num_kv_heads, KV_len, head_dim] -> [B, num_heads, KV_len, head_dim]
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # [B, num_heads, Q_len, head_dim] @ [B, num_heads, head_dim, KV_len] -> [B, num_heads, Q_len, KV_len]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # attn_weights: [B, num_heads, Q_len, KV_len] + [B, 1, Q_len, KV_len]
            attn_weights = attn_weights + attention_mask

        # attn_weights: [B, num_heads, Q_len, KV_len]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # [B, num_heads, Q_len, KV_len] @ [B, num_heads, KV_len, head_dim] -> [B, num_heads, Q_len, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)
        # [B, num_heads, Q_len, head_dim] -> [B, Q_len, num_heads, head_dim] -> [B, Q_len, num_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        # [B, Q_len, num_heads * head_dim] -> [B, Q_len, hidden_size]
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GemmaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        # hidden_states: [B, S, hidden_size]
        residual = hidden_states
        # norm: [B, S, hidden_size]
        hidden_states = self.input_layernorm(hidden_states)
        # attn: [B, S, hidden_size]
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        # residual: [B, S, hidden_size]
        hidden_states = residual + hidden_states

        residual = hidden_states
        # norm: [B, S, hidden_size]
        hidden_states = self.post_attention_layernorm(hidden_states)
        # mlp: [B, S, hidden_size]
        hidden_states = self.mlp(hidden_states)
        # residual: [B, S, hidden_size]
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id, dtype=config.dtype)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            # input_ids: [B, S] -> inputs_embeds: [B, S, hidden_size]
            inputs_embeds = self.embed_tokens(input_ids)

        # inputs_embeds: [B, S, hidden_size] * scalar -> [B, S, hidden_size]
        normalizer = math.sqrt(self.config.hidden_size)
        hidden_states = inputs_embeds * normalizer

        # hidden_states: [B, S, hidden_size] through num_layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # norm: [B, S, hidden_size]
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=config.dtype)

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Dict[str, Any]:
        # hidden_states: [B, S, hidden_size]
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        # lm_head(hidden_states): [B, S, vocab_size]
        logits = self.lm_head(hidden_states).float()

        result = {"logits": logits}
        if kv_cache is not None:
            result["kv_cache"] = kv_cache
        return result


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias=True,
            dtype=config.text_config.dtype,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        # image_features: [B, num_patches, vision_hidden_size]
        # output: [B, num_patches, projection_dim]
        return self.linear(image_features)


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.language_model = GemmaForCausalLM(config.text_config)
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else 0

    def tie_weights(self) -> None:
        self.language_model.tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.model.embed_tokens

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # image_features: [B, num_patches, D]
        # inputs_embeds: [B, seq_len, D]
        # input_ids: [B, seq_len]
        # attention_mask: [B, seq_len]
        batch_size, sequence_length = input_ids.shape
        _, _, embed_dim = image_features.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # scaled_image_features: [B, num_patches, D]
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)

        # final_embedding: [B, seq_len, D]
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=dtype, device=device)
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask = (input_ids == self.config.image_token_index)
        pad_mask = (input_ids == self.pad_token_id)

        # [B, seq_len] -> [B, seq_len, 1] -> [B, seq_len, D]
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        q_len = inputs_embeds.shape[1]
        if kv_cache is None or kv_cache.num_items() == 0:
            # causal_mask: [B, q_len, q_len] -> [B, 1, q_len, q_len]
            causal_mask = torch.full((batch_size, q_len, q_len), fill_value=0.0, dtype=dtype, device=device)
        else:
            assert q_len == 1, f"Expected q_len == 1 during decode, got {q_len}"
            kv_len = kv_cache.num_items() + q_len
            # causal_mask: [B, q_len, kv_len] -> [B, 1, q_len, kv_len]
            causal_mask = torch.full((batch_size, q_len, kv_len), fill_value=0.0, dtype=dtype, device=device)

        # [B, q_len, KV_len] -> [B, 1, q_len, KV_len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # position_ids: [B, 1]
            position_ids = attention_mask.cumsum(-1)[:, -1:].to(device)
        else:
            # position_ids: [B, seq_len]
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Dict[str, Any]:
        if attention_mask is None:
            # attention_mask: [B, seq_len]
            attention_mask = torch.ones_like(input_ids)

        # inputs_embeds: [B, seq_len, D]
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            # pixel_values: [B, 3, H, W]
            # selected_image_features: [B, num_patches, vision_hidden_size]
            selected_image_features = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
            # image_features: [B, num_patches, D]
            image_features = self.multi_modal_projector(selected_image_features)
            # inputs_embeds: [B, seq_len, D], attention_mask_causal: [B, 1, seq_len, seq_len], position_ids: [B, seq_len]
            inputs_embeds, attention_mask_causal, position_ids = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, kv_cache
            )
        else:
            batch_size, q_len = input_ids.shape
            dtype, device = inputs_embeds.dtype, inputs_embeds.device
            if kv_cache is not None and kv_cache.num_items() > 0:
                kv_len = kv_cache.num_items() + q_len
                # attention_mask_causal: [B, 1, 1, kv_len]
                attention_mask_causal = torch.full((batch_size, 1, q_len, kv_len), fill_value=0.0, dtype=dtype, device=device)
                # position_ids: [B, 1]
                position_ids = attention_mask.cumsum(-1)[:, -1:].to(device)
            else:
                attention_mask_causal = None
                # position_ids: [B, q_len]
                position_ids = attention_mask.cumsum(-1).to(device)

        # outputs["logits"]: [B, seq_len, vocab_size]
        outputs = self.language_model(
            attention_mask=attention_mask_causal,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs
