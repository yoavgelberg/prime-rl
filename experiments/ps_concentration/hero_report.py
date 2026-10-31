"""Plot complete saved hero batches without additional model evaluation."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def collect(run):
    run=Path(run)
    sources={}

    def captured(path):
        data=path.read_bytes()
        sources[str(path)]=hashlib.sha256(data).hexdigest()
        return json.loads(data)

    state=captured(run/'state.json')
    config=captured(run/'resolved_config.json')
    batch=config['boards']*config['group_size']
    committed=state['policy_version']
    excluded=set()
    recovery=run.parent/'recovery-receipt-a1.json'
    if recovery.exists():
        excluded.update(int(identity.split('-')[0][1:]) for identity in captured(recovery)['preserved_ids'])
    for path in run.glob('source-migration-*.json'):
        excluded.add(captured(path)['policy_version']+1)
    rows=[]
    for path in sorted((run/'update-metrics').glob('update-*.json')):
        point=captured(path);m=point['metrics'];update=m['update']
        if update>committed:
            continue
        if m['episodes']!=batch or m['training_episodes']!=update*batch:
            raise ValueError('Progress figures require complete batches at the configured global size')
        rows.append(dict(update=update,collection_policy_version=update-1,
            training_episodes=m['training_episodes'],mean_shaped_return=m['mean_return'],
            mean_unshaped_efficiency_return=m['mean_native_return'],mean_pairs=m['mean_pairs'],
            mean_responses=m['mean_attempts'],completed_games=m['completed_only_denominator'],
            format_errors=m['format_errors'],native_invalid_actions=m['native_invalid_actions'],
            generated_tokens=m['generated_tokens'],raw_transcript_tokens=m['raw_transcript_tokens'],
            seconds=m['seconds'],sampled_tokens_per_second_per_game=m['mean_per_game_tokens_per_second'],
            total_sampled_tokens_per_second=m['generated_tokens_per_second'],
            learner_peak_allocated_bytes=max([m['peak_training_allocated_bytes'],
                *[step['peak_allocated_bytes'] for step in point['steps']]]),
            kl_per_generated_token=m['kl_per_generated_token'],entropy_per_generated_token=m['entropy_per_generated_token'],
            gradient_norm=m['gradient_norm'],learning_rate=m['learning_rate'],
            recovery_or_source_continuation_cycle=update in excluded))
    if not rows or {r['update'] for r in rows}!=set(range(1,committed+1)):
        raise ValueError('Every committed update must have a complete saved metric record')
    return rows,config,sources,state


def report(run,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    run,output=Path(run),Path(output)
    rows,config,sources,state=collect(run)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'training-progress.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    x=[r['training_episodes'] for r in rows]
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    axes[0].plot(x,[r['mean_shaped_return'] for r in rows],'.-',label='Training reward')
    axes[0].plot(x,[r['mean_unshaped_efficiency_return'] for r in rows],'.-',label='Before error penalties')
    axes[0].set_ylabel('Mean efficiency reward');axes[0].legend(fontsize=8)
    axes[1].plot(x,[r['mean_pairs'] for r in rows],'.-');axes[1].set_ylim(0,8)
    axes[1].set_ylabel('Mean pairs found / 8')
    axes[2].plot(x,[r['mean_responses'] for r in rows],'.-');axes[2].set_ylim(0,96)
    axes[2].set_ylabel('Mean attempted responses')
    for ax in axes:
        ax.set_xlabel('Training episodes collected');ax.grid(alpha=.25)
    fig.suptitle(f'Matching Pairs 4×4 · PS W={config["window"]} · seed {config["seed"]} · sampled training')
    fig.savefig(output/'return-pairs-responses.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for field,ax,label,scale in [('seconds',axes[0],'Seconds / update',1),
        ('sampled_tokens_per_second_per_game',axes[1],'Sampled tokens/s/game',1),
        ('learner_peak_allocated_bytes',axes[2],'Peak learner allocated GiB',1/1024**3)]:
        clean=[r for r in rows if not r['recovery_or_source_continuation_cycle']]
        other=[r for r in rows if r['recovery_or_source_continuation_cycle']]
        ax.plot([r['update'] for r in clean],[r[field]*scale for r in clean],'.-',label='Ordinary cycle')
        if other:ax.scatter([r['update'] for r in other],[r[field]*scale for r in other],marker='x',color='gray',label='Recovery / continuation')
        ax.set_xlabel('Update');ax.set_ylabel(label);ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.savefig(output/'systems.png',dpi=180);plt.close(fig)
    latest=rows[-1]
    url=read(run/'wandb.json')['url'] if (run/'wandb.json').exists() else None
    lines=['# Matching Pairs PS-GRPO progress','',
        f'Complete through update {latest["update"]}: {latest["training_episodes"]:,} episodes. Latest scores were sampled from policy {latest["collection_policy_version"]}, before its associated optimizer step.',
        '',f'Latest batch: {latest["mean_pairs"]:.4f}/8 pairs, {latest["mean_responses"]:.2f} responses, training reward {latest["mean_shaped_return"]:+.6f}, efficiency reward before penalties {latest["mean_unshaped_efficiency_return"]:+.6f}.',
        '',f'Global batch {config["boards"]*config["group_size"]}, group {config["group_size"]}, seed {config["seed"]}, W={config["window"]}.',
        '', 'Efficiency reward is pairs/8 minus responses/96; the training reward also subtracts 0.02 per format error and native invalid action. RNG-Bench native score is pairs found. These are sampled training boards, not held-out greedy evaluation, a POPGym result, or evidence of causal textual memory.',
        '', 'Recovery and source-continuation cycles are marked separately in the timing plot. No speedup is inferred from those cycles.',
        '', '[Return, pairs, responses](return-pairs-responses.png) · [Systems](systems.png) · [CSV](training-progress.csv)']
    if url:lines.extend(['',f'[W&B]({url})'])
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    provenance=dict(utc=datetime.now(timezone.utc).isoformat(),run_dir=str(run),committed_update=latest['update'],
        inputs=sources,committed_state_snapshot=state,resolved_config_snapshot=config,
        plotting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        recovery_or_source_continuation_updates=[r['update'] for r in rows if r['recovery_or_source_continuation_cycle']])
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return dict(updates=len(rows),latest=latest,output=str(output))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(report(args.run_dir,args.output),indent=2))


if __name__=='__main__':
    main()
