import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from gemma_new.config import SiglipVisionConfig


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
            dtype=config.dtype,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim, dtype=config.dtype)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, C, H, W]
        # patch_embeds: [B, embed_dim, H / patch_size, W / patch_size]
        patch_embeds = self.patch_embedding(pixel_values)
        # [B, embed_dim, num_patches]
        embeddings = patch_embeds.flatten(2)
        # [B, num_patches, embed_dim]
        embeddings = embeddings.transpose(1, 2)
        # position_ids: [1, num_patches] -> position_embedding: [1, num_patches, embed_dim]
        # [B, num_patches, embed_dim] + [1, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, dtype=config.dtype)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, dtype=config.dtype)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, dtype=config.dtype)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, dtype=config.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, num_patches, embed_dim]
        batch_size, seq_len, _ = hidden_states.shape

        # [B, num_patches, embed_dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # [B, num_patches, embed_dim] -> [B, num_patches, num_heads, head_dim] -> [B, num_heads, num_patches, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # [B, num_heads, num_patches, head_dim] @ [B, num_heads, head_dim, num_patches] -> [B, num_heads, num_patches, num_patches]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        if self.dropout > 0.0:
            attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # [B, num_heads, num_patches, num_patches] @ [B, num_heads, num_patches, head_dim] -> [B, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)
        # [B, num_heads, num_patches, head_dim] -> [B, num_patches, num_heads, head_dim] -> [B, num_patches, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.embed_dim)
        # [B, num_patches, embed_dim]
        attn_output = self.out_proj(attn_output)
        return attn_output


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, dtype=config.dtype)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, dtype=config.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, num_patches, hidden_size]
        # fc1: [B, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # fc2: [B, num_patches, hidden_size]
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps, dtype=config.dtype)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps, dtype=config.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, num_patches, embed_dim]
        residual = hidden_states
        # norm1: [B, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states)
        # attn: [B, num_patches, embed_dim]
        hidden_states = self.self_attn(hidden_states)
        # residual: [B, num_patches, embed_dim]
        hidden_states = residual + hidden_states

        residual = hidden_states
        # norm2: [B, num_patches, embed_dim]
        hidden_states = self.layer_norm2(hidden_states)
        # mlp: [B, num_patches, embed_dim]
        hidden_states = self.mlp(hidden_states)
        # residual: [B, num_patches, embed_dim]
        hidden_states = residual + hidden_states

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # inputs_embeds: [B, num_patches, embed_dim]
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        # [B, num_patches, embed_dim]
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=config.dtype)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, C, H, W]
        # embeddings: [B, num_patches, embed_dim]
        hidden_states = self.embeddings(pixel_values)
        # encoder: [B, num_patches, embed_dim]
        hidden_states = self.encoder(hidden_states)
        # post_layernorm: [B, num_patches, embed_dim]
        last_hidden_state = self.post_layernorm(hidden_states)
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, C, H, W] -> output: [B, num_patches, embed_dim]
        return self.vision_model(pixel_values)
