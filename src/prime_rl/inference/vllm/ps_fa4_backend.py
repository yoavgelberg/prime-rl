import torch
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.utils.fa_utils import reshape_and_cache_flash
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import AttentionCGSupport

from prime_rl.trainer.ps_backward.paged import compact_pages, paged_ps_attention


class PSFA4MetadataBuilder(FlashAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.ALWAYS
    supports_update_block_table = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.aot_schedule = False
        self.window = kv_cache_spec.sliding_window
        if not self.window:
            raise ValueError("PS FA4 requires a sliding-window cache specification")
        rows = vllm_config.scheduler_config.max_num_seqs
        columns = (vllm_config.model_config.max_model_len + self.block_size - 1) // self.block_size
        self.ps_pages = torch.zeros((rows, columns), device=device, dtype=torch.int32)
        self.ps_fields = torch.zeros((5, rows), device=device, dtype=torch.int32)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(0, common_attn_metadata, fast_build=True)
        if metadata.prompt_lens is None:
            raise ValueError("PS requires the fork's per-request prompt-length metadata")
        rows, columns = metadata.block_table.shape
        rows = metadata.seq_lens.numel()
        metadata.ps_pages, metadata.ps_fields = compact_pages(
            metadata.block_table,
            metadata.seq_lens,
            metadata.query_start_loc,
            metadata.prompt_lens[:rows],
            self.block_size,
            self.window,
            output=self.ps_pages[:rows, :columns],
            fields=self.ps_fields[:, :rows],
        )
        return metadata


class PSFA4Impl(FlashAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.attn_type != AttentionType.DECODER or self.dcp_world_size != 1:
            raise ValueError("PS FA4 currently supports decoder attention without decode context parallelism")
        if self.alibi_slopes is not None or self.logits_soft_cap or self.sinks is not None:
            raise ValueError("PS FA4 requires unquantized scaled dot-product attention")
        if self.kv_cache_dtype != "auto":
            raise ValueError("PS FA4 currently requires BF16/FP16 KV caches")
        self.window = self.sliding_window[0] + 1
        self._initialized_value_cache = None

        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    def forward(
        self, layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None, output_block_scale=None
    ):
        if attn_metadata is None:
            return output
        if output_scale is not None or output_block_scale is not None:
            raise ValueError("Quantized attention outputs are unsupported")
        key_cache, value_cache = kv_cache.unbind(0)
        if self._initialized_value_cache != value_cache.data_ptr():
            value_cache.zero_()
            self._initialized_value_cache = value_cache.data_ptr()
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
        count = attn_metadata.num_actual_tokens
        paged_ps_attention(
            query[:count],
            key_cache,
            value_cache,
            attn_metadata.query_start_loc,
            attn_metadata.ps_pages,
            attn_metadata.ps_fields,
            self.window,
            attn_metadata.max_query_len,
            self.scale,
            output=output[:count],
        )
        return output


class PSFA4Backend(FlashAttentionBackend):
    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return PSFA4Impl

    @staticmethod
    def get_builder_cls():
        return PSFA4MetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [128]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability):
        return capability.major in (9, 10, 11)

    @classmethod
    def supports_attn_type(cls, attn_type):
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_sink(cls):
        return False
