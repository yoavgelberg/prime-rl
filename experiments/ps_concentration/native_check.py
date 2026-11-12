"""Check the unmodified HF generation interface on recorded screening inputs."""
import argparse
import gzip
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from .config import system_prompt
from .environment import completed_action, parse_action
from .rollout import ChatCodec


class ActionLine(StoppingCriteria):
    def __init__(self, tokenizer, start):
        self.tokenizer, self.start = tokenizer, start

    def __call__(self, input_ids, scores, **kwargs):
        return completed_action(self.tokenizer.decode(input_ids[0, self.start:], skip_special_tokens=True))


@torch.inference_mode()
def check(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    codec = ChatCodec(tokenizer)
    with gzip.open(args.episodes, 'rt') as handle:
        record = json.loads(next(handle))
    initial = tokenizer.decode(record['tokens'][len(codec.prefix):record['turns'][0]['start']],
                               skip_special_tokens=False)
    user_header, assistant_header = '<|im_start|>user\n', '<|im_end|>\n<|im_start|>assistant\n'
    assert initial.startswith(user_header) and initial.endswith(assistant_header)
    observation = initial[len(user_header):-len(assistant_header)]
    native = tokenizer.apply_chat_template([dict(role='system', content=system_prompt()),
                                            dict(role='user', content=observation)],
                                           tokenize=True, add_generation_prompt=True)
    assert native == record['tokens'][:record['turns'][0]['start']]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                attn_implementation='sdpa').to('cuda').eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    comparisons = []
    for turn in record['turns'][:2]:
        inputs = torch.tensor([record['tokens'][:turn['start']]], device='cuda')
        output = model.generate(inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                                max_new_tokens=256, pad_token_id=tokenizer.eos_token_id,
                                stopping_criteria=StoppingCriteriaList([ActionLine(tokenizer, inputs.shape[1])]))
        sampled = output[0, inputs.shape[1]:].tolist()
        text = tokenizer.decode(sampled, skip_special_tokens=True)
        stop = 'eos' if sampled[-1] in eos_ids else 'action_line' if completed_action(text) else 'length'
        action, error = parse_action(text, 52, stop_condition=stop)
        comparisons.append(dict(step=turn['step'], input_tokens=inputs.shape[1], native_tokens=sampled,
                                native_text=text, native_stop=stop, native_action=action,
                                native_format_error=error, recorded_text=turn['text'],
                                recorded_stop=turn['stop_condition'], recorded_format_error=turn['format_error'],
                                identical_tokens=sampled == record['tokens'][turn['start']:turn['end']]))
    return dict(scope='Two fixed screening inputs; interface diagnostic only, no task score',
                board_seed=record['board_seed'], native_chat_template_equal=True,
                model=args.model, dtype='bfloat16', gpu=torch.cuda.get_device_name(), comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--episodes', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = check(args)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
