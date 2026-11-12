"""Independent native game actor with in-place policy weight updates."""

from prime_rl.inference.vllm.ps_streaming_worker import PSStreamingWorker


class HeroInferenceWorker(PSStreamingWorker):
    def ps_hero_load_weights(self, path, policy_version):
        import torch

        from prime_rl.inference.vllm import ps_interactive
        from prime_rl.inference.vllm.ps_capture import _store
        from prime_rl.trainer.ps_backward.qwen3 import runner_model

        if _store.active or ps_interactive._cohort is not None:
            raise RuntimeError("Update actor weights only after all previous games are released")
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if state["policy_version"] != policy_version:
            raise ValueError("Actor weight version differs from the rollout batch")
        loaded = runner_model(self.model_runner).load_weights(state["model"].items())
        torch.cuda.synchronize()
        _store.policy_version = policy_version
        return dict(policy_version=policy_version, loaded_parameters=len(loaded), recording=False)
