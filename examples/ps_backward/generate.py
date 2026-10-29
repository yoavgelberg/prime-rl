"""Native AsyncLLM FA4-PS generation, optionally with streaming score gradients.

Inputs are JSONL rows containing prompt_ids. No reward or optimizer update runs.
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from prime_rl.trainer.ps_backward.config import StreamConfig


async def run(args):
    from vllm.config import AttentionConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    from prime_rl.inference.vllm.ps_fa4 import register

    register()
    records = [json.loads(row) for row in Path(args.prompts).read_text().splitlines() if row.strip()]
    if not records:
        raise ValueError("Provide at least one prompt")
    optimized = (
        dict(
            joint_qkv=True,
            direct_weight_gradients=True,
            weight_gradient_backend="cublas",
            compact_kv=True,
            kv_cache=True,
            fine_grained_events=True,
            gpu_head_staging=True,
            stage_scheduling=True,
            copy_programs=8,
            gpu_stages_in_flight=4,
            backward_batch_tokens=4096,
        )
        if args.optimized
        else {}
    )
    options = StreamConfig(
        **optimized,
        window=args.window,
        chunk_tokens=args.chunk,
        ring_tokens=2 * args.chunk,
        max_requests=args.requests,
        fp32_attention=not args.native_attention_gradients,
    )
    prefix = max(len(record["prompt_ids"]) for record in records)
    if args.streaming and prefix > options.chunk_tokens:
        raise ValueError("Increase --chunk to cover the longest prompt")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")

    extra = dict(worker_cls=args.worker, additional_config={"ps_streaming": asdict(options)}) if args.streaming else {}
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            revision=args.revision,
            dtype="bfloat16",
            skip_tokenizer_init=True,
            tensor_parallel_size=1,
            max_model_len=max(args.length, args.warm_length) + prefix,
            max_num_seqs=args.requests,
            max_num_batched_tokens=max(2048, args.chunk),
            block_size=128,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            kv_cache_memory_bytes=8 * 1024**3,
            attention_config=AttentionConfig(backend="CUSTOM"),
            hf_overrides={
                "use_sliding_window": True,
                "sliding_window": args.window,
                "max_position_embeddings": max(args.length, args.warm_length) + prefix,
            },
            **extra,
        )
    )
    save("config.json", vars(args))

    async def generate(index, length, phase):
        prompt = records[index % len(records)]["prompt_ids"]
        request = f"{phase}-{index}"
        params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=length,
            ignore_eos=True,
            seed=42 + index,
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        async for response in engine.generate({"prompt_token_ids": prompt, "cache_salt": request}, params, request):
            tokens = list(response.outputs[0].token_ids)
        if args.streaming and phase != "native-warm":
            await engine.collective_rpc("ps_stream_request_finished", args=(request,))
        return dict(request_id=request, prefix_length=len(prompt), input_ids=[*prompt, *tokens])

    async def cohort(length, phase):
        return await asyncio.gather(
            *(generate(i, length if i % 2 else 3 * length // 4, phase) for i in range(args.requests))
        )

    try:
        await cohort(args.warm_length, "native-warm")
        await engine.reset_prefix_cache()
        if args.streaming:
            await engine.collective_rpc("ps_stream_begin", kwargs={"policy_version": 0, "warmup": True})
            await cohort(args.warm_length, "backward-warm")
            save("warmup.json", await engine.collective_rpc("ps_stream_finish"))
            save("graphs.json", await engine.collective_rpc("ps_stream_prepare_graphs"))
            await engine.reset_prefix_cache()
            await engine.collective_rpc("ps_stream_reset")
        started = time.perf_counter()
        traces = await cohort(args.length, "measure")
        generation_seconds = time.perf_counter() - started
        save(
            "generation.json",
            dict(seconds=generation_seconds, tokens=sum(len(t["input_ids"]) - t["prefix_length"] for t in traces)),
        )
        save("traces.json", traces)
        if args.streaming:
            report = await engine.collective_rpc("ps_stream_finish")
            save("streaming.json", report)
            if args.audit:
                save("audit.json", await engine.collective_rpc("ps_stream_audit", args=(traces,)))
        print(json.dumps({"output": str(output), "generation_seconds": generation_seconds}), flush=True)
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--warm-length", type=int, default=8192)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument(
        "--optimized", action="store_true", help="Select the combined B200-qualified streaming settings"
    )
    parser.add_argument(
        "--native-attention-gradients", action="store_true", help="Hopper-compatible, not long-trace qualified"
    )
    parser.add_argument("--worker", default="prime_rl.inference.vllm.ps_streaming_worker.PSStreamingWorker")
    parser.add_argument("--audit", action="store_true", help="Requires the test-only AuditWorker")
    args = parser.parse_args()
    if args.optimized and not args.streaming:
        parser.error("--optimized requires --streaming")
    # The pinned PS fork reads these settings at import time.
    os.environ["SWF"] = str(args.window)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
