import json
from pathlib import Path
import time

import torch


def restore_adamw(optimizer,state):
    optimizer.load_state_dict(state)
    assert all(not g.get('capturable') and not g.get('fused') for g in optimizer.param_groups)
    for values in optimizer.state.values():
        values['step']=values['step'].cpu()


def inference_state(policy):
    direct={id(m.weight) for m in policy.modules() if isinstance(m,torch.nn.Embedding)}
    rounded={f'{name}.weight' for name,m in policy.named_modules()
             if isinstance(m,torch.nn.Linear) and id(m.weight) not in direct}
    return {name:(value.detach().bfloat16() if name in rounded else value.detach()).cpu()
            for name,value in policy.state_dict().items()}


def save_checkpoint(path,policy,tokenizer,metadata):
    path.mkdir(parents=True,exist_ok=False)
    policy.save_pretrained(path,safe_serialization=True,state_dict=inference_state(policy))
    tokenizer.save_pretrained(path)
    metadata=dict(metadata,storage='BF16 projection operands; FP32 embeddings, normalization and biases; exact effective policy')
    (path/'experiment.json').write_text(json.dumps(metadata,indent=2))


def retain_selected_and_latest(output,best_update,latest_update):
    keep={best_update,latest_update}
    for path in sorted(Path(output).glob('checkpoint-*')):
        update=int(path.name.removeprefix('checkpoint-'))
        if update in keep or (path/'weights-retired.json').exists():
            continue
        removed=[]
        for binary in path.glob('model*.safetensors'):
            removed.append(dict(name=binary.name,bytes=binary.stat().st_size))
            binary.unlink()
        (path/'weights-retired.json').write_text(json.dumps(dict(
            reason='Scratch capacity: retain selected and latest effective-policy weights; preserve all checkpoint metadata',
            removed=removed,selected_update=best_update,latest_update=latest_update,retired_unix=time.time()),indent=2))
