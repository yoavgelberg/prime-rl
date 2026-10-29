"""Run the released RNG-Bench text game with a local full-context HF policy."""
import argparse
import gzip
import json
from pathlib import Path
import re
import sys
import time
import types

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


class AnswerLine(StoppingCriteria):
    def __init__(self, tokenizer, start, thinking):
        self.tokenizer, self.start, self.thinking = tokenizer, start, thinking

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0,self.start:], skip_special_tokens=True)
        if self.thinking:
            if '</think>' not in text:
                return False
            text = text.rsplit('</think>',1)[-1]
        return re.search(r'(?:^|\n)ANSWER \d+[ \t]*\n',text) is not None


class LocalPolicy:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                        attn_implementation='sdpa').to('cuda').eval()
        self.calls = []

    @torch.inference_mode()
    def chat(self, messages):
        text_messages = []
        for message in messages:
            content = message['content']
            if isinstance(content, list):
                assert all(part['type'] == 'text' for part in content)
                content = '\n'.join(part['text'] for part in content)
            text_messages.append(dict(role=message['role'], content=content))
        inputs = self.tokenizer.apply_chat_template(text_messages, tokenize=True, add_generation_prompt=True,
                                                    return_tensors='pt').to('cuda')
        started = time.perf_counter()
        kwargs = dict(do_sample=self.args.sample, max_new_tokens=self.args.max_new_tokens,
                      pad_token_id=self.tokenizer.eos_token_id)
        thinking = self.tokenizer.decode(inputs[0,-8:],skip_special_tokens=False).rstrip().endswith('<think>')
        if getattr(self.args,'stop_at_answer',False):
            kwargs['stopping_criteria'] = StoppingCriteriaList([AnswerLine(self.tokenizer,inputs.shape[1],thinking)])
        if self.args.sample:
            kwargs.update(temperature=getattr(self.args,'temperature',.6),
                          top_p=getattr(self.args,'top_p',.95), top_k=getattr(self.args,'top_k',20))
        output = self.model.generate(inputs, attention_mask=torch.ones_like(inputs), **kwargs)
        sampled = output[0, inputs.shape[1]:].tolist()
        text = self.tokenizer.decode(sampled, skip_special_tokens=True)
        self.calls.append(dict(input_ids=inputs[0].tolist(), sampled_ids=sampled,
                               seconds=time.perf_counter()-started, output=text))
        if thinking:
            if '</think>' in text:
                reasoning, content = text.rsplit('</think>',1)
            else:
                reasoning, content = text, ''
            return dict(content=content.strip(), reasoning=reasoning)
        return dict(content=text, reasoning=None)


def run(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(11)
    source = Path(args.source)
    lock = json.loads((source/'SOURCE.json').read_text())
    sys.path.insert(0, str(source/'1_matching_pairs_new'))
    client = LocalPolicy(args)
    shim = types.ModuleType('model_presets')
    shim.make_client = lambda *a, **kw: client
    shim.parse_grid_size = lambda value: tuple(map(int, value.split('x')))
    sys.modules['model_presets'] = shim
    from modes.single_normal import run_one_game
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(model=args.model, upstream=lock, rows=args.rows, cols=args.cols,
                    max_responses_per_pair=None if args.unbounded else 5, max_retries=2, max_new_tokens=args.max_new_tokens,
                    sample=args.sample, temperature=args.temperature if args.sample else 0., full_context=True,
                    text_transport='Native text content parts joined with newlines; local HF generation',
                    scope='Task-qualification diagnostic, not a reproduction of published model scores')
    (output/'config.json').write_text(json.dumps(metadata, indent=2))
    rows = []
    for seed in range(1_000_000, 1_000_000+args.count):
        client.calls = []
        result = run_one_game(model='local', rows=args.rows, cols=args.cols, num_cards=None,
                              seed=seed, render='text', theme=None, assets_dir=str(source/'1_matching_pairs_new/assets'),
                              cell_size=64, out_dir=output/f'seed-{seed}',
                              max_responses=sys.maxsize if args.unbounded else 5*args.rows*args.cols//2,
                              max_retries=2, cot_enabled=True)
        assert result.get('error') is None, result
        with gzip.open(output/f'seed-{seed}-tokens.json.gz', 'wt') as handle:
            json.dump(client.calls, handle)
        row = dict(seed=seed, **result, matched_fraction=result['score']/result['total_pairs'],
                   generated_tokens=sum(len(c['sampled_ids']) for c in client.calls),
                   inference_seconds=sum(c['seconds'] for c in client.calls))
        rows.append(row)
        (output/'summary.json').write_text(json.dumps(dict(episodes=len(rows),
            mean_matched_fraction=sum(r['matched_fraction'] for r in rows)/len(rows),
            completion_rate=sum(r['done'] for r in rows)/len(rows), episodes_detail=rows), indent=2))
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--cols', type=int, default=4)
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=.7)
    parser.add_argument('--top-p', type=float, default=.8)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--unbounded', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
