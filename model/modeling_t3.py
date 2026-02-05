from abc import abstractmethod
import math
from typing import Callable, List, Optional, Sequence, Tuple, Dict
import logging
# from dataclasses import fields

import torch
from torch import nn
from torch.nn import functional as F
from peft import LoraConfig, get_peft_model, TaskType

# from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.Qwen3.modeling_qwen3 import Qwen3ForCausalLM
from model.LLaDA.configuration_llada import BlockType, LLaDAConfig, ActivationCheckpointingStrategy, ModelConfig as LLaDAModelConfig
from model.LLaDA.modeling_llada import (
    Activation,
    BufferCache,
    Dropout,
    LayerNormBase,
    LayerNorm,
    LLaDABlock,
    LLaDABlockGroup,
    LLaDAOutput,
    ModuleType,
    RotaryEmbedding,
    LLaDAModelLM, 
    _non_meta_init_device,
    activation_checkpoint_function,
    alibi_attention_bias,
    ensure_finite_,
    get_causal_attention_bias,
    init_weights,
    )

log = logging.getLogger(__name__)

def manual_init_talk_model(model, config):
    print("Applying manual initialization to talk_model...")
    init_std = getattr(config, "init_std", 0.02)

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        print("Initializing fc ...")
        nn.init.normal_(model.fc.weight, std=init_std)
        if model.fc.bias is not None:
            nn.init.zeros_(model.fc.bias)

    # Transformer blocks depth scaling
    blocks = model.transformer.blocks
    for layer_idx, block in enumerate(blocks):
        scaled_std = init_std / math.sqrt(2 * (layer_idx + 1))
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                if "attn_out" in name or "ff_out" in name:
                    nn.init.trunc_normal_(
                        module.weight, mean=0.0, std=scaled_std,
                        a=-3 * scaled_std, b=3 * scaled_std
                    )
                elif "proj" in name:
                    nn.init.trunc_normal_(
                        module.weight, mean=0.0, std=init_std,
                        a=-3 * init_std, b=3 * init_std
                    )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

def print_param_devices(model: torch.nn.Module):
    print("=== Parameter devices ===")
    for name, param in model.named_parameters():
        print(f"{name:80s} -> {param.dtype} {param.device} {param.shape}")

class LLaDAFuseBlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: LLaDAModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)

        # Activation function.
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        self.attn_out = nn.Linear(
            2 * config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
        )

        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config, self.__cache)

        self.flash_attn_func = None
        if config.flash_attention:
            try:
                from flash_attn import flash_attn_func  # type: ignore

                self.flash_attn_func = flash_attn_func
            except ModuleNotFoundError:
                pass

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(
            self.config,
            self.attn_out,
            d=self.config.d_model,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        if self.flash_attn_func is not None and attn_mask is None:
            r = self.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=False
            )
            return r.transpose(1, 2)
        else:
            # torch's sdpa doesn't support GQA, so we're doing this
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

            # Modify: MDM set causal to False, and with no attn_mask.
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        replace_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None: #self.q_norm: None, self.k_norm: None
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        # self.config.n_heads: 32
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is not None: 
            past_key, past_value = layer_past
            if replace_position is None:
                k = torch.cat((past_key, k), dim=-2)
                v = torch.cat((past_value, v), dim=-2)
            else:
                # k shape is [B, n_kv_h, selected_length, hs]
                # replace_position shape is [B, L], where L contains 0s and 1s, 0 means no replacement, 1 means replace, with selected_length number of 1s
                # past_key shape is [B, n_kv_h, L, hs]
                # Replace selected_length number of 1s in past_key with k
                
                # Handle batched replace_position correctly
                B = replace_position.shape[0]
                for batch_idx in range(B):
                    # Get indices for this batch
                    batch_replace_indices = replace_position[batch_idx].nonzero(as_tuple=True)[0]
                    if len(batch_replace_indices) > 0:
                        # Replace positions in past_key and past_value for this batch
                        past_key[batch_idx, :, batch_replace_indices] = k[batch_idx, :, :len(batch_replace_indices)]
                        past_value[batch_idx, :, batch_replace_indices] = v[batch_idx, :, :len(batch_replace_indices)]
                
                k = past_key
                v = past_value

        present = (k, v) if use_cache else None #present: None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if self.config.rope:
            # Apply rotary embeddings.
            if replace_position is None:
                q, k = self.rotary_emb(q, k)
            else:
                # For batched replace_position, use the maximum position across all batches
                max_replace_pos = replace_position.nonzero(as_tuple=True)[1].max() + 1 if replace_position.any() else key_len
                q, k = self.rotary_emb(q, k, max_replace_pos)

        if attention_bias is not None:
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=False,
        )
        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), present

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: LLaDAModelConfig, cache: BufferCache) -> LLaDABlock:
        if config.block_type == BlockType.llama:
            return LLaDALlamaFuseBlock(layer_id, config, cache)
        # elif config.block_type == BlockType.sequential:
        #     return LLaDASequentialBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")

class LLaDALlamaFuseBlock(LLaDABlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `LLaDASequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: LLaDAModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.input_norm = LayerNorm.build(config)
        self.hidden_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(
            2 * config.d_model, q_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.k_proj = nn.Linear(
            2 * config.d_model, k_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.v_proj = nn.Linear(
            2 * config.d_model, v_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )
        # new add
        self.up_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.input_norm.reset_parameters()
        self.hidden_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.up_proj, d=self.config.d_model, layer_id=None)  # new add

    def forward(
        self,
        inputs_emb: torch.Tensor,
        hidden_states: torch.Tensor, 
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        replace_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)
        x = hidden_states
        inputs_emb = self.input_norm(inputs_emb) #x:torch.Size([2, 168, 4096])
        hidden_states = self.hidden_norm(hidden_states)
        x_normed = torch.concat([inputs_emb, hidden_states], dim=-1)
        q = self.q_proj(x_normed) #q:torch.Size([2, 168, 4096])
        k = self.k_proj(x_normed) #k:torch.Size([2, 168, 4096])
        v = self.v_proj(x_normed) #v:torch.Size([2, 168, 4096])

        # attention_bias: None
        # layer_past: None
        # use_cache: False
        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache
                # ,replace_position=replace_position
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache
                                        # ,replace_position=replace_position
                                        )

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x, x_up = self.ff_proj(x), self.up_proj(x) # new add
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = x * x_up # new add
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


class TalkModel(nn.Module):
    def __init__(self, config: LLaDAModelConfig, init_params: bool=True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        # Validate config.
        if self.config.alibi and self.config.flash_attention:
            raise Exception("ALiBi is currently not supported with FlashAttention")

        if self.config.alibi and self.config.rope:
            raise Exception("ALiBi and RoPE are mutually exclusive")

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise Exception("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise Exception("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it
                
        self.transformer = nn.ModuleDict(
            dict(
                # wte=wte,
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )
        blocks = [LLaDAFuseBlock.build(0, config, self.__cache)] + [LLaDABlock.build(i, config, self.__cache) for i in range(1, config.n_layers)]
        if self.config.block_group_size > 1:
            block_groups = [
                LLaDABlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
        # if not config.weight_tying:
        #     self.transformer.update(
        #         {
        #             "ff_out": nn.Linear(
        #                 config.d_model,
        #                 config.embedding_size or config.vocab_size,
        #                 bias=config.include_bias,
        #                 device=config.init_device,
        #             )
        #         }
        #     )

        self.fc = nn.Linear(config.d_model * 3, config.d_model)
        self.rps_norm = LayerNorm.build(config)  # same LN type as model uses
        self.rps_gate = nn.Linear(config.d_model, config.d_model)  # or to 1 for scalar gate
        self.rps_post_norm = LayerNorm.build(config)  # in __init__

        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params:# and self.config.init_device != "meta":
            manual_init_talk_model(self, self.config)
        self.__num_fwd_flops: Optional[int] = None
        # Warm up cache.
        if self.config.alibi:
            get_causal_attention_bias(self.__cache, config.max_sequence_length, _non_meta_init_device(config))
            self.get_alibi_attention_bias(config.max_sequence_length, _non_meta_init_device(config))


    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        if self.config.block_group_size != 1:
            for block_group in self.transformer.block_groups:
                block_group.set_activation_checkpointing(strategy)
        else:
            for block in self.transformer.blocks:
                block.set_activation_checkpointing(strategy)

    # @property
    # def device(self) -> torch.device:
    #     device: torch.device = self.transformer.emb_drop.device  # type: ignore
    #     if device.type == "meta":
    #         return _non_meta_init_device(self.config)
    #     else:
    #         return device
        
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device       



    def reset_parameters(self):
        log.info("Initializing model parameters...")
        # Top-level embeddings / linear layers.
        # if not self.is_wte_shared:
        #     init_weights(
        #         self.config,
        #         self.transformer.wte,
        #         std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0,
        #         type_of_module=ModuleType.emb,
        #     )
        if hasattr(self.transformer, "wpe"):
            init_weights(self.config, self.transformer.wpe, type_of_module=ModuleType.emb)  # type: ignore

        # Top-level layer norm.
        self.transformer.ln_f.reset_parameters()  # type: ignore

        # Output weights.
        # if hasattr(self.transformer, "ff_out"):
        #     init_weights(self.config, self.transformer.ff_out, type_of_module=ModuleType.final_out)  # type: ignore

        # Let the blocks handle themselves.
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()

    def get_alibi_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (alibi_bias := self.__cache.get("alibi_attention_bias")) is not None and alibi_bias.shape[
            -1
        ] >= seq_len:
            if alibi_bias.device != device:
                alibi_bias = alibi_bias.to(device)
                self.__cache["alibi_attention_bias"] = alibi_bias
            return alibi_bias
        with torch.autocast(device.type, enabled=False):
            alibi_bias = alibi_attention_bias(seq_len, self.config, device)
        self.__cache["alibi_attention_bias"] = alibi_bias
        return alibi_bias

    def fuse_representation(self, x):
        return self.fc(x)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        input_repres: torch.FloatTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        replace_position: Optional[torch.Tensor] = None,
        rps_residual: Dict = None,
    ):
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """
        # Add Basic MDM Model config check
        assert not self.config.alibi, "Alibi length extrapolation is not supported for MDM."
        assert self.config.rope, "Rope must be used in Llama-Encoder for MDM."
        # assert (past_key_values is None and not use_cache), "The kvcache is not suppotred for MDM."

        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)
        
        """
        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore
        """
        batch_size, seq_len = input_embeddings.size()[:2]

        x = input_embeddings
        if self.config.input_emb_norm:
            x = x * (self.config.d_model**0.5)

        if not (self.config.alibi or self.config.rope):
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None and 0.0 in attention_mask:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        else:
            attention_mask = None

        # Merge attention mask with attention bias.
        if (
            attention_bias is not None
            or attention_mask is not None
            or self.config.alibi
            # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
            # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
            # scores correctly.
            or past_key_values is not None
        ):
            if attention_bias is None and self.config.alibi:
                attention_bias = get_causal_attention_bias(
                    self.__cache, past_length + seq_len, x.device
                ) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
            elif attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                # it can produce NaNs.
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                if (
                    (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
                        and block_idx % 2 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
                        and block_idx % 3 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
                        and block_idx % 4 == 0
                    )
                ):
                    if block_idx == 0:
                        x, cache = self._activation_checkpoint_fn(
                            block, x, hidden_states=input_repres, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache,
                            # replace_position=replace_position
                        )
                    else:
                        # shape: (batch_size, seq_len, d_model)
                        x, cache = self._activation_checkpoint_fn(
                            block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache,
                            # replace_position=replace_position
                        )
                else:
                    if block_idx == 0:
                        x, cache = block(x, hidden_states=input_repres, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache,
                                        #  replace_position=replace_position
                        )
                    else:
                        # shape: (batch_size, seq_len, d_model)
                        x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache,
                                        #  replace_position=replace_position
                                         )
                        
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )
                x, cache = block_group(
                    x, attention_bias=attention_bias, layers_past=layers_past, use_cache=use_cache
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        # if self.config.weight_tying:
        #     logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        # else:
        #     logits = self.transformer.ff_out(x)  # type: ignore
        # if self.config.scale_logits:
        #     logits.mul_(1 / math.sqrt(self.config.d_model))

        if rps_residual["enabled"] and output_hidden_states:
            rps_in = self.rps_norm(input_repres)
            x_last = all_hidden_states[-1]  # post ln_f
            gate = torch.sigmoid(self.rps_gate(x_last))
            rps_next = rps_in + rps_residual["eta"] * gate * x_last
            # keep in same space
            rps_next = self.rps_post_norm(rps_next)
            all_hidden_states[-1] = rps_next

        return LLaDAOutput(logits=None, attn_key_values=attn_key_values, hidden_states=tuple(all_hidden_states) if output_hidden_states else None)  # type: ignore[arg-type]


class T3Model(nn.Module):
    def __init__(
        self,
        config: dict,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict = "auto",
        train: bool = True,
        think_dev1: str = "cuda:1",
        think_dev2: str = "cuda:2",
        talk_dev: str = "cuda:0"
    ):
        super().__init__()
        
        self.is_training_mode = train
        self.think_dev1 = think_dev1
        self.think_dev2 = think_dev2
        self.talk_dev = talk_dev
        self.device = self.think_dev1

        device_map = None

        # Length
        self.length = config["length"]

        # Think model
        self.architecture = self._register(config["pretrained_model_name_or_path"])
        self.think_model = self._load_model(config["pretrained_model_name_or_path"],dtype=dtype, device_map=device_map)
        if config["lora"]["enabled"]:
            self._add_lora(config.get("lora", {}))
            self.think_model_root = self.think_model.base_model.model
        else:
            self.think_model_root = self.think_model
        # self.think_model.to(self.think_dev1)
        self.think_model_root.model.set_pipeline(
            split_points=(config["lora"]["start_layer"],),  # e.g. 20
            devices=(self.think_dev1, self.think_dev2),     # 2 devices for 1 split
        )
        self.think_model_root.model.move_blocks()
        self.think_model_root.model.set_recorded_hidden_index(config["mix_indexes"])

        # Talk model
        talk_config = LLaDAModelConfig(**config["talk_model"])
        talk_config.init_device = "cpu" 
        self.talk_model = TalkModel(talk_config)
        self.talk_model.to(self.talk_dev)

        # structure
        self.rps_residual = config["rps_residual"]

        # For train / inference efficient
        if self.think_dev1 != self.talk_dev or self.think_dev2 != self.talk_dev:
            # copy embed
            if self.architecture == "Qwen3":
                self.talk_embed_weight = self.think_model_root.model.embed_tokens.weight.detach()
            if self.architecture == "LLaDA":
                self.talk_embed_weight = self.think_model_root.model.transformer.wte.weight.detach() 
            # copy lm head
            self.talk_lm_head_bias = None
            if self.architecture == "Qwen3":
                self.talk_lm_head_weight = self.think_model_root.lm_head.weight.detach()
                if self.think_model_root.lm_head.bias is not None:
                    self.talk_lm_head_bias = self.think_model_root.lm_head.bias.detach()                
            if self.architecture == "LLaDA":
                self.talk_lm_head_weight = self.think_model_root.model.transformer.ff_out.weight.detach()
                if self.think_model_root.model.transformer.ff_out.bias is not None:
                    self.talk_lm_head_bias = self.think_model_root.model.transformer.ff_out.bias.detach()
            # assign to talk model
            self.talk_embed_weight = self.talk_embed_weight.to(self.talk_dev)
            self.talk_lm_head_weight = self.talk_lm_head_weight.to(self.talk_dev)
            if self.talk_lm_head_bias is not None:
                self.talk_lm_head_bias = self.talk_lm_head_bias.to(self.talk_dev)
        else: 
            if self.architecture == "Qwen3":
                self.talk_embed_weight = self.think_model_root.model.embed_tokens.weight
                self.talk_lm_head_weight = self.think_model_root.lm_head.weight
                self.talk_lm_head_bias = self.think_model_root.lm_head.bias
            if self.architecture == "LLaDA":
                self.talk_embed_weight = self.think_model_root.model.transformer.wte.weight
                self.talk_lm_head_weight = self.think_model_root.model.transformer.ff_out.weight
                self.talk_lm_head_bias = self.think_model_root.model.transformer.ff_out.bias

        # Actions for limited memory resource
        if self.architecture == "LLaDA":
            if hasattr(self.think_model_root.model.transformer, "ff_out") and self.think_dev2 != self.talk_dev:
                del self.think_model_root.model.transformer.ff_out
            self.prune_llada_last_n_blocks(config["prune_last_n_layer"])
            from model.LLaDA.configuration_llada import ActivationCheckpointingStrategy
            self.think_model_root.model.set_activation_checkpointing(
                ActivationCheckpointingStrategy.whole_layer
            )

        # Handling Device
        # if self.is_training_mode:
        # self.think_model.to(self.think_dev)
        # self.talk_model.to(self.talk_dev)
        # self.talk_embed_weight = self.talk_embed_weight.to(self.talk_dev)
        # self.talk_lm_head_weight = self.talk_lm_head_weight.to(self.talk_dev)
        # if self.talk_lm_head_bias is not None:
        #     self.talk_lm_head_bias = self.talk_lm_head_bias.to(self.talk_dev)
        
        dt  = next(self.think_model.parameters()).dtype
        self.talk_model.to(dtype=dt)
        # print_param_devices(self.think_model)
        # print_param_devices(self.talk_model)
    
    def prune_llada_last_n_blocks(self, n_remove: int):
        """
        Remove the last `n_remove` transformer blocks.
        This reduces parameter memory and compute.

        WARNING: This changes the model architecture; load checkpoints carefully.
        """
        assert n_remove >= 0
        n_layers = self.think_model_root.config.n_layers
        assert n_remove < n_layers, f"Can't remove {n_remove} blocks from {n_layers} layers"

        new_n_layers = n_layers - n_remove

        if self.think_model_root.config.block_group_size == 1:
            # transformer.blocks is a ModuleList
            self.think_model_root.model.transformer.blocks = nn.ModuleList(list(self.think_model_root.model.transformer.blocks[:new_n_layers]))
        else:
            # blocks are packed into block_groups
            g = self.think_model_root.config.block_group_size
            assert new_n_layers % g == 0, (
                f"new_n_layers={new_n_layers} must be divisible by block_group_size={g}. "
                f"Choose n_remove accordingly."
            )
            n_groups = new_n_layers // g
            self.think_model_root.model.transformer.block_groups = nn.ModuleList(list(self.think_model_root.model.transformer.block_groups[:n_groups]))

        # Update config + any logic that depends on layer count
        self.think_model_root.config.n_layers = new_n_layers

        # Reset cache-dependent counters if any
        self.__num_fwd_flops = None

    def _register(self, model_name):
        if "LLaDA" in model_name:
            return "LLaDA"
        if "Qwen3" in model_name:
            return "Qwen3"
    
    def _load_model(self, pretrained_model_name_or_path: str, dtype: torch.dtype, device_map):
        if "Qwen3" in pretrained_model_name_or_path:
            return Qwen3ForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                device_map=device_map,
                dtype=dtype,
            )

        if "LLaDA" in pretrained_model_name_or_path:
            return LLaDAModelLM.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                device_map=device_map,
                dtype=dtype,
            )

        raise ValueError(f"Unknown model type: {pretrained_model_name_or_path}")

    def _add_lora(self, lora_cfg: dict):
        base_targets = lora_cfg.get("target_modules", {})[self.architecture]
        start = lora_cfg.get("start_layer", 20)

        # infer total layers (works for most HF causal LMs)
        n_layers = getattr(self.think_model.config, "num_hidden_layers", None)
        if n_layers is None:
            # fallback: try common attribute names
            n_layers = getattr(self.think_model.config, "n_layer", None)

        if n_layers is None:
            raise ValueError("Can't infer num layers from model config; please pass n_layers manually.")
        
        layer_ids = list(range(start, n_layers))
        target_modules = []
        for n, p in self.think_model.named_parameters():
            for m in base_targets:
                if m in n:
                    for lid in layer_ids:
                        if str(lid) in n:
                            target_modules.append(f"{lid}.{m}")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=target_modules,
            bias=lora_cfg.get("bias", "none"),   # "none" | "all" | "lora_only"
            inference_mode=False,
            use_dora=lora_cfg.get("use_dora", False)
        )

        self.think_model = get_peft_model(self.think_model, peft_config)
        self.think_model.print_trainable_parameters()

    def embed_think(self, input_ids):
        if self.architecture == "Qwen3":
            inputs_embeds = self.think_model_root.model.embed_tokens(input_ids)
        if self.architecture == "LLaDA":
            inputs_embeds = self.think_model_root.model.transformer.wte(input_ids)
        return inputs_embeds

    def embed_talk(self, input_ids):
        return F.embedding(input_ids, self.talk_embed_weight)
    
    def lm_head_talk(self, x):
        return F.linear(x, self.talk_lm_head_weight, self.talk_lm_head_bias)

    def run_think_model(
        self,
        inputs_embeds: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,  # LLaDA
        position_ids: torch.LongTensor | None = None,  # Qwen3
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,  # Qwen3
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        ):

        """
        return past_key_values, representation
        """
        if self.architecture == "Qwen3":
            outputs = self.think_model_root.model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                return_dict=True,
            )
        if self.architecture == "LLaDA":
            outputs = self.think_model_root.model(
                input_ids=None,
                input_embeddings=inputs_embeds,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                past_key_values=past_key_values,
                use_cache=use_cache,
                last_logits_only=last_logits_only,
                output_hidden_states=output_hidden_states,
                )

        if self.architecture == "Qwen3":
            past_key_values = outputs.past_key_values
        if self.architecture == "LLaDA":
            past_key_values = outputs.attn_key_values

        return outputs.hidden_states, past_key_values
    
    def get_hidden_representation(self, hidden_states):
        """
        Use EAGLE3 logic: fuse first, mid, last hidden representation
        """
        assert len(hidden_states) == 3
        dev_out = hidden_states[-1].device
        rps = torch.concat([hidden_states[0].to(dev_out), hidden_states[1].to(dev_out), hidden_states[-1]], dim=-1)

        # one_third = len(hidden_states) // 3
        # two_third = 2 * len(hidden_states) // 3
        # rps = torch.concat([hidden_states[one_third], hidden_states[two_third], hidden_states[-1]], dim=-1)
        # mid = len(hidden_states) // 2
        # rps = torch.concat([hidden_states[1], hidden_states[mid], hidden_states[-1]], dim=-1)
        return rps
    
    def run_talk_model(
            self,
            input_ids: torch.LongTensor,
            input_repres: torch.FloatTensor,
            input_embeddings: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            attention_bias: Optional[torch.Tensor] = None,
            past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            last_logits_only: bool = False,
            output_hidden_states: Optional[bool] = None,
            replace_position: Optional[torch.Tensor] = None,
                       ):
        
        # fuse rps back to size d
        if input_repres.shape[-1] == 3 * self.talk_model.config.d_model:
            input_repres = self.talk_model.fuse_representation(input_repres)
        
        outputs = self.talk_model(
                input_ids=input_ids,
                input_repres=input_repres,
                input_embeddings=input_embeddings,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                past_key_values=past_key_values,
                use_cache=use_cache,
                last_logits_only=last_logits_only,
                output_hidden_states=output_hidden_states,
                replace_position=replace_position,
                rps_residual=self.rps_residual
                )
            
        return outputs
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_repres: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: torch.LongTensor | None = None,  # Qwen3
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        cache_position: Optional[Cache] = None,
        # block_mask: Optional[torch.Tensor] = None
        ):
        
        if inputs_repres is None:
            inputs_embeds = self.embed_think(input_ids)

            hidden_states, past_key_values = self.run_think_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache or False,
                cache_position=cache_position,
                output_hidden_states=output_hidden_states,
            )

            inputs_repres = self.get_hidden_representation(hidden_states)
            logits = None
        else:
            if inputs_embeds is None:
                inputs_embeds = self.embed_talk(input_ids)

            outputs = self.run_talk_model(
                input_ids=None,
                input_repres=inputs_repres,
                input_embeddings=inputs_embeds,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                past_key_values=past_key_values,
                use_cache=use_cache,
                last_logits_only=False,
                output_hidden_states=output_hidden_states,
            )

            inputs_repres = outputs.hidden_states[-1]
            logits = self.lm_head_talk(inputs_repres)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=inputs_repres,
        )
        
def list_module_name_hints(model: nn.Module, contains=("q", "k", "v", "o", "proj")):
    hits = set()
    for name, module in model.named_modules():
        if any(s in name for s in contains):
            # keep leaf-ish modules that are likely Linear layers
            if isinstance(module, torch.nn.Linear):
                hits.add(name.split(".")[-1])
    print("Candidate leaf module names:", sorted(hits))

def smoke_test(t3, full_len=24, talk_len=4, batch=2):
    dev = next(t3.think_model.parameters()).device
    vocab_guess = 32000

    # full sequence for think
    full_ids = torch.randint(0, vocab_guess, (batch, full_len), device=dev)
    full_mask = torch.ones((batch, full_len), device=dev)

    out_think = t3(
        input_ids=full_ids,
        attention_mask=full_mask,
        inputs_repres=None,
        output_hidden_states=True,
        use_cache=False,
    )
    # You currently return only CausalLMOutputWithPast, so we need to recompute repres:
    # (Better: return repres too, but keeping your interface)
    repres = out_think.hidden_states
    print("repres:", repres.shape)  # [B, full_len, 3D]

    # talk only on masked span tokens (toy)
    talk_ids = full_ids[:, -talk_len:]               # pretend last 4 are masked span tokens
    # IMPORTANT: if talk takes only 4 tokens, repres must match 4 tokens too
    # You said you want retrieval outside: so slice repres accordingly
    talk_repres = repres[:, -talk_len:, :]           # [B, talk_len, 3D]

    out_talk = t3(
        input_ids=talk_ids,
        attention_mask=torch.ones((batch, talk_len), device=dev),
        inputs_repres=talk_repres,
        output_hidden_states=True,
        use_cache=False,
    )
    print("logits:", None if out_talk.logits is None else out_talk.logits.shape)

if __name__ == "__main__":
    import json
    
    with open("model/config.json", "r") as f:
        cfg = json.load(f)
    t3 = T3Model(cfg, use_lora=True)
    smoke_test(t3)