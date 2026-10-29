"""Inference-only worker for persistent native games under either attention mask."""

from vllm.v1.worker.gpu_worker import Worker

from .ps_interactive import InteractiveWorkerMixin, install_sampling_hook


class GameWorker(InteractiveWorkerMixin, Worker):
    def load_model(self):
        install_sampling_hook()
        super().load_model()
