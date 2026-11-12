# Continue the W1024 policy toward 25 turns

The user extended the exploratory W1024 run past update 448, targeting approximately 25 individual card flips while preserving completion of all eight pairs. Update 448 is a continuation boundary. Preserve the saved model, Adam and per-rank RNG states, the original frozen reference, learning rate 1e-6, the warmup already completed at 328, temperature 0.8, group eight, batch 128, exact KL and entropy coefficients, controller and neutral prompt. No further warmup is needed at an allocation or budget boundary within this phase.

Use an operational horizon of 1024, extending it if needed under the same authorized performance goal. The shared inference metadata accepts horizons within the million-board training pool. Actual training still uses 16 new boards per update without replacement. The implementation change adds only horizon validation, a graceful stop request and final inference export; transformer, attention, sampling, objective and optimizer computations remain unchanged.

To finish after a complete optimizer step, write `RUN_DIR/stop-after-update.json` atomically with `after_update`, the exact current training `identity`, and a descriptive `reason`. The coordinator validates the request, retains every sampled-token loss, saves a full final model/Adam/RNG checkpoint and exports the final BF16 policy. The request is optional; an absent request leaves the configured horizon and allocation deadline in force. Check the final full-state and export preservation receipts before releasing the allocation. A resumed launcher should avoid restarting an already fulfilled request.

Assess the approximate 25-turn target over eight consecutive complete batches (1024 sampled games), including failures and all invalid/format attempts. A working operational criterion is mean responses at most 25.5, at least 99% completion and mean pairs at least 7.99. These are sampled-training stopping diagnostics, not held-out results or evidence that scratch text causes the memory. Keep the prompt unchanged and inspect old-reveal utilization, retained associations and later separate-seed interventions.

The previously prepared fixed-448 evaluation plan remains preserved but deferred. Freeze the final policy and updated evaluation plan before inspecting fresh held-out boards. Continue logging to the same W&B run and save rolling full checkpoints every five updates.

After the training-only turn target is met and the final full checkpoint and inference export are verified, freeze the actual stopped policy before inspecting fresh boards:

```bash
uv run python -m experiments.ps_concentration.hero_goal25_final --run-dir RUN_DIR --output FINAL_DIR --gpus 4 --source-root CAPTURED_SOURCE --source-sha256 ARCHIVE_SHA256
```

This command validates the completed training marker, graceful-stop identity and every batch in the declared stopping window, then records immutable parent/continued configuration snapshots. The resulting `FINAL_DIR/plan.json` preserves the fresh7M/8M board ranges and separate4M intervention collection, uses the actual stopped version, and defaults to PS evaluations. Run it with `hero_post320_final run --plan FINAL_DIR/plan.json --gpus 4 --phase ps`; the same module's `memory` command runs the later common-cold intervention branches. Generate tables and figures using `hero_post320_report --final-plan FINAL_DIR/plan.json --output REPORT_DIR`. The original fixed448 plan remains preserved as historical preparation. These commands do not authorize GPU evaluations while the user has prioritized continued training.
