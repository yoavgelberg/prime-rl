"""Register the FA4-PS backend; select it with attention_backend="CUSTOM"."""


def register():
    from vllm.attention.backends.registry import AttentionBackendEnum, register_backend

    register_backend(AttentionBackendEnum.CUSTOM, "prime_rl.inference.vllm.ps_fa4_backend.PSFA4Backend")
