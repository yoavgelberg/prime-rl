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
                mean_loss_error = abs(float(stream.loss) - row['negative_logp_sum']) / stream.loss_tokens
                if not mean_loss_error <= 1e-6:
                    raise ValueError('Native captured policy loss differs from its actual sampled likelihoods')
                row['native_capture_mean_loss_error'] = mean_loss_error
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

    def ps_train_save_pending(self, path, metadata):
        from pathlib import Path
        from prime_rl.trainer.ps_backward.checkpoint import save_pending_gradients

        trainer = self.ps_trainer
        if not trainer.materialized or (self.ps_kl_coefficient != 0 and self.ps_kl_ready):
            raise RuntimeError('Save pending task gradients after materialization and before reference KL')
        rank = self.vllm_config.parallel_config.data_parallel_rank
        with torch.inference_mode(False):
            return save_pending_gradients(Path(path) / f'rank-{rank}.pt', trainer.model,
                self.ps_train_records, policy_version=trainer.policy_version, metadata=metadata)

    def ps_train_restore_pending(self, path, expected_metadata):
        from pathlib import Path
        from prime_rl.trainer.ps_backward.checkpoint import restore_pending_gradients
        from .ps_capture import _store

        trainer = self.ps_trainer
        if trainer.materialized or trainer.records or _store.live is not None:
            raise RuntimeError('Restore pending gradients before admitting new trajectories')
        rank = self.vllm_config.parallel_config.data_parallel_rank
        with torch.inference_mode(False):
            state = restore_pending_gradients(Path(path) / f'rank-{rank}.pt', trainer.model,
                policy_version=trainer.policy_version, expected_metadata=expected_metadata)
        rows = state['records']
        if len(rows) != trainer.config.global_batch_size // trainer.world_size:
            raise ValueError('Pending gradients require the original GPU replica count')
        if any(row['rank'] != rank or row['policy_version'] != trainer.policy_version
               or not row.get('native_capture_mean_loss_error', float('inf')) <= 1e-6 for row in rows):
            raise ValueError('Pending trajectory rank, policy version or native-loss certificate differs')
        trainer.records = {row['episode_id']: dict(board=row['board_index'], reward=row['training_return'],
            generated_tokens=row['generated_tokens'], negative_logp_sum=row['negative_logp_sum']) for row in rows}
        if len(trainer.records) != len(rows):
            raise ValueError('Duplicate pending trajectory identity')
        self.ps_train_records = rows
        trainer.materialized, self.ps_kl_ready = True, False
        return dict(policy_version=trainer.policy_version, episodes=len(rows), rank=rank)

    def ps_train_reference_kl(self, backend, options):
        if self.ps_kl_coefficient == 0:
            if not self.ps_trainer.materialized:
                raise RuntimeError('Materialize native task gradients before the optimizer step')
            self.ps_kl_ready = True
            return dict(kl_sum=0., normalized_kl=0., generated_tokens=sum(
                row['generated_tokens'] for row in self.ps_train_records), disabled=True,
                reference_kl_seconds=0., parity=dict(mean_abs=None, p99=None, status='DISABLED'))
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

    def ps_train_save(self, path, metadata, *, inference_path=None, tokenizer_source=None):
        from pathlib import Path
        import torch.distributed as dist
        from vllm.distributed.parallel_state import get_dp_group
        from prime_rl.trainer.ps_backward.checkpoint import save_inference, save_snapshot
        from .ps_capture import _store

        trainer = self.ps_trainer
        if trainer.materialized or trainer.records or (_store.live is not None and not _store.live.closed):
            raise RuntimeError('Snapshots require a complete optimizer boundary with no in-flight trajectories')
        parallel = get_dp_group()
        rank = parallel.rank_in_group
        with torch.inference_mode(False):
            rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(self.device))
            if parallel.world_size > 1:
                rngs = [None] * parallel.world_size
                dist.all_gather_object(rngs, rng, group=parallel.device_group)
                from prime_rl.trainer.ps_backward.optimizer_serialization import fast_optimizer_serialization
                with fast_optimizer_serialization():
                    trainer.optimizer.consolidate_state_dict(to=0)
            else:
                rngs = [rng]
            receipt = None
            if rank == 0:
                receipt = save_snapshot(path, trainer.model, trainer.optimizer.state_dict(),
                    policy_version=trainer.policy_version, outcome_config=trainer.config, rngs=rngs,
                    metadata=dict(metadata, world_size=parallel.world_size))
                if inference_path is not None:
                    if tokenizer_source is None:
                        raise ValueError('Saving inference weights requires the pinned tokenizer source')
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
                    save_inference(Path(inference_path), trainer.model.base, tokenizer,
                        policy_version=trainer.policy_version, metadata=metadata)
            if parallel.world_size > 1:
                trainer.optimizer._all_state_dicts.clear()
                dist.barrier(group=parallel.device_group)
        return dict(policy_version=trainer.policy_version, rank=rank, snapshot=receipt)

    def ps_train_restore(self, path, expected_metadata):
        import hashlib
        from vllm.distributed.parallel_state import get_dp_group
        from prime_rl.trainer.ps_backward.checkpoint import restore_snapshot
        from .ps_capture import _store

        trainer = self.ps_trainer
        if _store.live is not None or trainer.records or trainer.materialized:
            raise RuntimeError('Restore the optimizer before beginning any rollout cohort')
        parallel = get_dp_group()
        with torch.inference_mode(False):
            state = restore_snapshot(path, trainer.model, trainer.optimizer,
                outcome_config=trainer.config, expected_metadata=expected_metadata)
            version = int(state['policy_version'])
            trainer.policy_version = version
            trainer.model._ps_update_version = version
            _store.policy_version = version
            if parallel.rank_in_group < len(state['rngs']):
                rng = state['rngs'][parallel.rank_in_group]
                torch.set_rng_state(rng['cpu'])
                torch.cuda.set_rng_state(rng['cuda'], device=self.device)
                rng_source = 'saved per-rank CPU/CUDA states'
            else:
                identity = f"recorded-grpo-layout-resume:{state['metadata']['training_seed']}:{version}:{parallel.rank_in_group}"
                seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little') % (2**63 - 1)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                rng_source = 'derived for a newly added GPU rank; per-request generation seeds remain layout independent'
            update_inference_weights(trainer.model.base, runner_model(self.model_runner))
        torch.cuda.synchronize()
        return dict(policy_version=version, rank=parallel.rank_in_group, rng_source=rng_source,
            prior_world_size=len(state['rngs']), current_world_size=parallel.world_size,
            metadata=state['metadata'])
