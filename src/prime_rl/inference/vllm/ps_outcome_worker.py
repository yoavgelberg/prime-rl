"""Worker-side bridge from recorded PS derivatives to synchronous GRPO updates."""

import gc
import importlib
import time

import torch

from prime_rl.trainer.ps_backward.outcome import OutcomeConfig, OutcomeTrainer
from prime_rl.trainer.ps_backward.qwen3 import load_backward_model, runner_model, update_inference_weights


class OutcomeWorkerMixin:
    def ps_train_initialize(self, config):
        from torch.distributed.optim import ZeroRedundancyOptimizer
        from vllm.distributed.parallel_state import get_dp_group
        from .ps_capture import _store

        if hasattr(self, 'ps_trainer') or _store.live is not None:
            raise RuntimeError('Initialize the optimizer once, before recording the first cohort')
        optimizer_config = config['optimizer']
        outcome = OutcomeConfig(**config['outcome'])
        if _store.config.loss_denominator != outcome.denominator:
            raise ValueError('Streaming score gradients must use the same fixed denominator as GRPO')
        parallel = get_dp_group()
        with torch.inference_mode(False), torch.enable_grad():
            model = load_backward_model(self.model_runner, _store.config.window)
            options = dict(lr=outcome.learning_rate, betas=tuple(optimizer_config['betas']),
                eps=optimizer_config['epsilon'], weight_decay=0., fused=True)
            if parallel.world_size > 1:
                optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW,
                    process_group=parallel.device_group, parameters_as_bucket_view=True, **options)
            else:
                optimizer = torch.optim.AdamW(model.parameters(), **options)
            self.ps_trainer = OutcomeTrainer(model, optimizer, outcome, group=parallel.device_group,
                world_size=parallel.world_size)
        self.ps_kl_coefficient = config['kl_coefficient']
        self.ps_kl_ready = False
        self.ps_reduced_requests = set()
        self.ps_board_identities = {}
        self.ps_train_records = []
        return dict(world_size=parallel.world_size, policy_version=0,
            optimizer_state_sharding=parallel.world_size > 1,
            parameters=sum(p.numel() for p in model.parameters()))

    def ps_train_reduce(self, *, capture_graphs=True):
        from .ps_capture import _store

        live, trainer = _store.live, self.ps_trainer
        if not live.closed or live.warmup or _store.policy_version != trainer.policy_version:
            raise RuntimeError('Reduce only a complete cohort at its unchanged rollout policy')
        rows = self.ps_game_finish()
        audits = []
        with torch.inference_mode(False), torch.no_grad():
            for row in rows:
                identity = row['episode_id']
                if identity in self.ps_reduced_requests:
                    raise ValueError('A sampled trajectory may contribute its gradient only once')
                matches = [key for key in live.streams if key == identity or key.startswith(identity + '-')]
                if len(matches) != 1:
                    raise ValueError('Native game must identify exactly one recorded trajectory')
                stream = live.streams[matches[0]]
                if stream.loss_tokens != row['generated_tokens']:
                    raise ValueError('Direct loss count differs from sampled assistant-token count')
                board = row['board_index']
                board_identity = row['board_identity_sha256']
                previous = self.ps_board_identities.setdefault(board, board_identity)
                if previous != board_identity:
                    raise ValueError('Rollouts within one GRPO group did not reset to the identical board')
                trainer.add(identity, board, stream.gradients, row['training_return'],
                    policy_version=trainer.policy_version, generated_tokens=row['generated_tokens'],
                    negative_logp_sum=row['negative_logp_sum'])
                audits.append(dict(episode_id=identity, direct_loss_tokens=stream.loss_tokens,
                    rollout_negative_logp_sum=row['negative_logp_sum'], recorded_negative_logp_sum=float(stream.loss),
                    mean_sum_error=abs(float(stream.loss) - row['negative_logp_sum']) / stream.loss_tokens))
                self.ps_reduced_requests.add(identity)
                self.ps_train_records.append(row)
            graphs = None
            if capture_graphs and any(stage.graph is None for stage in live.graph_cache.stages.values()):
                with torch.enable_grad():
                    graphs = live.graph_cache.capture()
                live.graph_cache.require_ready = False
            # Capture may write discarded private gradients; the CPU reductions above are already immutable.
            for stream in live.streams.values():
                stream.gradients.clear()
            live.model.zero_grad(set_to_none=True)
        return dict(episodes=len(rows), audits=audits, graphs=graphs,
            cpu_group_buffers_bytes=sum(value.numel() * value.element_size() for group in trainer.groups.values()
                for values in (group.gradient_sum, group.reward_gradient_sum) for value in values.values()),
            gpu_allocated_bytes=torch.cuda.memory_allocated())

    def ps_train_materialize(self):
        from .ps_capture import _store

        if _store.live is None or not _store.live.closed or _store.active:
            raise RuntimeError('Drain generation and recorded backward before materializing an update')
        with torch.inference_mode(False):
            result = self.ps_trainer.materialize()
        tape = _store.live.host_tape
        if tape is not None:
            tape.reset(lambda: None)
            tape.layer_pools.clear()
            tape.head_pool.buffers.clear()
            _store.live.host_tape = None
            gc.collect()
        self.ps_kl_ready = self.ps_kl_coefficient == 0
        return dict(**result, generation_host_pools_released=True,
            gpu_allocated_bytes=torch.cuda.memory_allocated())

    def ps_train_reference_kl(self, backend, options):
        if not self.ps_trainer.materialized or self.ps_kl_ready:
            raise RuntimeError('Reference KL must be applied once after task-gradient materialization')
        module, name = backend.rsplit('.', 1)
        apply = getattr(importlib.import_module(module), name)
        options = dict(options)
        options['output'] = str(options['output']) + f"/rank-{self.vllm_config.parallel_config.data_parallel_rank}"
        with torch.inference_mode(False), torch.enable_grad():
            result = apply(self.ps_trainer.model, self.ps_train_records,
                coefficient=self.ps_kl_coefficient, outcome_config=self.ps_trainer.config,
                initial_policy=self.ps_trainer.policy_version == 0, **options)
        self.ps_kl_ready = True
        return result

    def ps_train_step(self):
        from .ps_capture import _store

        if not self.ps_kl_ready:
            raise RuntimeError('Apply the configured frozen-reference KL before an optimizer step')
        started = time.perf_counter()
        with torch.inference_mode(False):
            result = self.ps_trainer.step()
            update_inference_weights(self.ps_trainer.model.base, runner_model(self.model_runner))
        torch.cuda.synchronize()
        _store.policy_version = self.ps_trainer.policy_version
        self.ps_kl_ready = False
        self.ps_reduced_requests.clear()
        self.ps_board_identities.clear()
        self.ps_train_records.clear()
        return dict(**result, optimizer_and_weight_refresh_seconds=time.perf_counter() - started)
