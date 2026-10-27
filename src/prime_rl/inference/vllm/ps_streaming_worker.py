"""Thin Qwen3 adapter between native vLLM sampling and recorded PS backward."""

import math

import torch
from vllm.v1.worker.gpu_worker import Worker

from prime_rl.trainer.ps_backward.config import StreamConfig
from prime_rl.trainer.ps_backward.qwen3 import load_backward_model


class PSStreamingWorker(Worker):
    def load_model(self):
        config = self.vllm_config
        hf = config.model_config.hf_config
        if hf.model_type != "qwen3" or getattr(hf, "attention_dropout", 0):
            raise ValueError("The recording adapter requires a dense Qwen3 model without dropout")
        if config.parallel_config.tensor_parallel_size != 1 or config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("The recording adapter requires TP=PP=1")
        if config.model_config.quantization is not None or config.lora_config is not None:
            raise ValueError("Quantized and LoRA inference weights need separate recording adapters")
        if config.speculative_config is not None:
            raise ValueError("Recorded score gradients require ordinary autoregressive sampling")
        if config.model_config.dtype != torch.bfloat16:
            raise ValueError("The recording adapter stores BF16 inference activations")
        options = StreamConfig(**config.additional_config["ps_streaming"])
        options.validate_hardware(
            self.device, query_heads=hf.num_attention_heads, kv_heads=hf.num_key_value_heads, head_dim=hf.head_dim
        )
        if options.max_requests > config.scheduler_config.max_num_seqs:
            raise ValueError("max_num_seqs must cover the configured streaming cohort")
        from .ps_capture import install

        install()
        super().load_model()

    def ps_stream_begin(self, policy_version=0, warmup=False):
        from .ps_capture import _store
        from .ps_live import LiveBackward

        if _store.live is not None:
            raise RuntimeError("Use ps_stream_reset after draining this worker's cohort")
        _store.policy_version = policy_version
        with torch.inference_mode(False), torch.enable_grad():
            model = load_backward_model(self.model_runner, _store.config.window)
            _store.live = LiveBackward(_store, self.model_runner, model, warmup=warmup)
        _store.active = True
        return _store.report()

    def ps_stream_request_finished(self, request_id):
        from .ps_capture import _store

        matches = [key for key in _store.requests if key == request_id or key.startswith(request_id + "-")]
        if len(matches) != 1:
            raise ValueError("Request ID must resolve to exactly one recorded rollout")
        _store.live.finish_request(matches[0])

    def ps_stream_finish(self):
        from .ps_capture import _store

        result = _store.live.finish()
        _store.active = False
        return result

    def ps_stream_prepare_graphs(self):
        from .ps_capture import _store

        if not _store.live.closed or not _store.live.warmup:
            raise RuntimeError("Drain a warmup cohort before CUDA graph capture")
        with torch.inference_mode(False), torch.enable_grad():
            return _store.live.graph_cache.capture()

    def ps_stream_reset(self):
        """Reuse buffers/graphs after warmup at the same frozen policy version."""
        from .ps_capture import _store
        from .ps_live import LiveBackward

        previous = _store.live
        if not previous.closed:
            raise RuntimeError("Drain the cohort before resetting recording slots")
        _store.requests.clear()
        _store.last_requests = None
        for tensor in (_store.request_slots, _store.head_slots, _store.token_slots):
            tensor.fill_(-1)
        _store.overflow.zero_()
        previous.model.zero_grad(set_to_none=True)
        previous.streams.clear()
        with torch.inference_mode(False), torch.enable_grad():
            _store.live = LiveBackward(
                _store,
                self.model_runner,
                previous.model,
                graph_cache=previous.graph_cache,
                host_tape=previous.host_tape,
                cuda_stream=previous.cuda_stream,
            )
        _store.active = True
        torch.cuda.reset_peak_memory_stats()
        return _store.report()

    def ps_stream_export_gradients(self, advantages, policy_version):
        """Return the batch-mean loss gradient on CPU; never apply an optimizer update.

        Keys are the exact request IDs returned by ps_stream_finish. Each scalar
        multiplies that trajectory's mean negative-log-probability gradient.
        Export/transport time is additional to generation and backward timing.
        """
        from .ps_capture import _store

        live = _store.live
        if not live.closed or live.warmup or policy_version != _store.policy_version:
            raise ValueError("Export requires a drained non-warmup cohort at the frozen policy version")
        if not advantages or advantages.keys() != live.streams.keys():
            raise ValueError("Provide one scalar advantage for every completed request")
        if any(not math.isfinite(value) for value in advantages.values()):
            raise ValueError("Advantages must be finite")
        gradients = {}
        with torch.inference_mode(False), torch.no_grad():
            for name in dict(live.model.named_parameters()):
                value = torch.zeros_like(live.streams[next(iter(advantages))].gradients[name])
                for request, advantage in advantages.items():
                    value.add_(live.streams[request].gradients[name], alpha=advantage / len(advantages))
                gradients[name] = value.cpu()
        return gradients
