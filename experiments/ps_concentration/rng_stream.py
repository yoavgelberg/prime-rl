"""Paired full/PS native-thinking evaluation on RNG-Bench Matching Pairs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import types

import torch
from transformers import AutoTokenizer

from .policy import load_model, StreamingQwen


class TranscriptCapacity(RuntimeError):
    pass


def text_content(message):
    value = message['content']
    if isinstance(value, str):
        return value
    assert all(part['type'] == 'text' for part in value)
    return '\n'.join(part['text'] for part in value)


def generation_seed(board):
    digest = hashlib.sha256(f'rng-thinking-stream-v1:11:{board}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') % (2**63-1)


class StreamingClient:
    def __init__(self, base, tokenizer, args):
        self.base, self.tokenizer, self.args = base, tokenizer, args
        self.stream = None

    def reset(self, seed, output):
        self.stream = None
        self.tokens, self.roles, self.old_logps, self.turns = [], [], [], []
        self.seen = 0
        self.seed = seed
        self.output = output
        self.rng = torch.Generator(device=self.base.device).manual_seed(generation_seed(seed))
        self.started = time.perf_counter()
        self.pinned = None
        torch.cuda.reset_peak_memory_stats()

    def _check_capacity(self, count):
        if len(self.tokens)+count > self.args.transcript_cap:
            raise TranscriptCapacity('Native context/transcript capacity reached; censor, never score as native failure')

    def _append(self, tokens, role):
        self._check_capacity(len(tokens))
        self.tokens.extend(tokens)
        self.roles.extend([role]*len(tokens))
        self.old_logps.extend([None]*len(tokens))
        logits = None
        for token in tokens:
            logits = self.stream.step(torch.tensor([token],device=self.base.device))
        return logits

    def _sample(self, logits):
        raw = logits.float()[0]
        scaled = raw/self.args.temperature
        values, indices = scaled.topk(self.args.top_k)
        probs = values.softmax(-1)
        probs.masked_fill_(probs.cumsum(-1)-probs >= self.args.top_p,0.)
        probs /= probs.sum()
        selected = torch.multinomial(probs,1,generator=self.rng)
        token = indices[selected][0]
        logp = raw[token]-raw.logsumexp(-1)
        token_value, old_logp = torch.stack((token.float(),logp)).tolist()
        return int(token_value),old_logp

    @torch.inference_mode()
    def chat(self, messages):
        new = [dict(role=m['role'],content=text_content(m)) for m in messages[self.seen:]]
        if self.stream is None:
            assert new[0]['role']=='system'
            self.system = new.pop(0)
            self.prefix = self.tokenizer.apply_chat_template([self.system],tokenize=True,
                                                              add_generation_prompt=False)
            assert len(self.prefix)<=512,len(self.prefix)
            rendered = self.tokenizer.apply_chat_template(
                [self.system,*new],tokenize=True,add_generation_prompt=True)
            assert rendered[:len(self.prefix)]==self.prefix
            assert self.tokenizer.decode(rendered[-8:],skip_special_tokens=False).rstrip().endswith('<think>')
            self.stream = StreamingQwen(self.base,1,len(self.prefix),self.args.window,
                                        self.args.attention,self.args.transcript_cap)
            self._check_capacity(len(rendered))
            logits = self.stream.prefill(torch.tensor([rendered],device=self.base.device))
            self.tokens.extend(rendered)
            self.roles.extend(['prefix']*len(self.prefix)+['environment_template']*(len(rendered)-len(self.prefix)))
            self.old_logps.extend([None]*len(rendered))
            self.pinned = [k[:,:,:len(self.prefix)].clone() for k in self.stream.keys]
            marker = 'UNIQUE_CONTENT_MARKER_8037'
            rendered_close = self.tokenizer.apply_chat_template(
                [dict(role='assistant',content=marker)],tokenize=False,add_generation_prompt=False)
            self.close = self.tokenizer.encode(rendered_close.split(marker,1)[1],add_special_tokens=False)
        else:
            assert new and all(m['role']=='user' for m in new)
            closing = self.close[1:] if self.tokens[-1]==self.close[0] else self.close
            self._append(closing,'template')
            rendered = self.tokenizer.apply_chat_template([self.system,*new],tokenize=True,
                                                          add_generation_prompt=True)
            assert rendered[:len(self.prefix)]==self.prefix
            logits = self._append(rendered[len(self.prefix):],'environment_template')
        native_eos = self.base.generation_config.eos_token_id
        eos = set(native_eos if isinstance(native_eos,list) else [native_eos])
        sampled, logps = [], []
        turn_started = time.perf_counter()
        start = len(self.tokens)
        while True:
            self._check_capacity(1)
            token, logp = self._sample(logits)
            sampled.append(token)
            logps.append(logp)
            self.tokens.append(token)
            self.roles.append('assistant')
            self.old_logps.append(logp)
            logits = self.stream.step(torch.tensor([token],device=self.base.device))
            if token in eos:
                stop = 'eos'
                break
            if token == self.tokenizer.encode('\n',add_special_tokens=False)[0]:
                text = self.tokenizer.decode(sampled,skip_special_tokens=True)
                if '</think>' in text and re.search(r'(?:^|\n)Action:\s*[a-z][A-Z][ \t]*\n',text.rsplit('</think>',1)[-1]):
                    stop = 'action_line'
                    break
        text = self.tokenizer.decode(sampled,skip_special_tokens=True)
        reasoning, final = text.rsplit('</think>',1) if '</think>' in text else (text,'')
        self.turns.append(dict(start=start,end=len(self.tokens),sampled_ids=sampled,old_logps=logps,
                               stop_condition=stop,seconds=time.perf_counter()-turn_started,
                               reasoning=reasoning,content=final.strip()))
        self.seen = len(messages)+1
        for keys, pinned in zip(self.stream.keys,self.pinned,strict=True):
            assert torch.equal(keys[:,:,:len(self.prefix)],pinned)
        assert self.args.attention=='full' or self.stream.capacity==len(self.prefix)+self.args.window
        progress=dict(seed=self.seed,attention=self.args.attention,turns=len(self.turns),
                      raw_transcript_tokens=len(self.tokens),generated_tokens=sum(len(t['sampled_ids']) for t in self.turns),
                      seconds=time.perf_counter()-self.started,latest_action=final.strip())
        (self.output/f'progress-{self.args.start_seed}.json').write_text(json.dumps(progress,indent=2))
        with gzip.open(self.output/f'seed-{self.seed}-turns.jsonl.gz','at') as handle:
            handle.write(json.dumps(self.turns[-1])+'\n')
        print(json.dumps(progress),flush=True)
        return dict(content=final.strip(),reasoning=reasoning)

    def save(self, row):
        record=dict(row,tokens=self.tokens,roles=self.roles,
                    loss_mask=[r=='assistant' for r in self.roles],old_logps=self.old_logps,
                    positions=list(range(len(self.tokens))),turns=self.turns,
                    prefix_length=len(self.prefix),window=self.args.window,attention=self.args.attention,
                    generation_seed=generation_seed(self.seed),thinking_retained=True,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    allocated_kv_bytes=sum(k.numel()*k.element_size()*2 for k in self.stream.keys),
                    active_kv_bytes=self.stream.cache_bytes)
        with gzip.open(self.output/f'seed-{self.seed}-tokens.json.gz','wt') as handle:
            json.dump(record,handle)


def run(args):
    source = Path(args.source)
    sys.path.insert(0,str(source/'1_matching_pairs_new'))
    base = load_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    client = StreamingClient(base,tokenizer,args)
    shim = types.ModuleType('model_presets')
    shim.make_client = lambda *a,**kw: client
    shim.parse_grid_size = lambda value: tuple(map(int,value.split('x')))
    sys.modules['model_presets'] = shim
    from modes.single_normal import run_one_game
    from common import prompts
    prompts.OUTPUT_FORMAT_NO_COT = '''Output format:
After thinking, end your final response with exactly one action line:
Action: <coordinate, e.g. cD>
'''
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    metadata = dict(**vars(args),model_revision=Path(args.model).name,
                    upstream=json.loads((source/'SOURCE.json').read_text()),
                    system_prompt=prompts.build_single_system(cot_enabled=False),
                    thinking_retained=True,max_new_tokens=None,max_responses=None,
                    arithmetic='bf16_products_attention_fp32_residual_and_projection_accumulation',
                    scope='Base qualification; no RL or SFT',gpu=torch.cuda.get_device_name())
    (output/f'config-{args.start_seed}.json').write_text(json.dumps(metadata,indent=2))
    rows=[]
    for seed in range(args.start_seed,args.start_seed+args.count):
        receipt=output/f'seed-{seed}-result.json'
        if receipt.exists():
            rows.append(json.loads(receipt.read_text()))
            continue
        client.reset(seed,output)
        try:
            result=run_one_game(model='local',rows=4,cols=4,num_cards=None,seed=seed,render='text',
                theme=None,assets_dir=str(source/'1_matching_pairs_new/assets'),cell_size=64,
                out_dir=output/f'seed-{seed}',max_responses=sys.maxsize,max_retries=2,cot_enabled=False)
            assert result.get('error') is None,result
            row=dict(seed=seed,**result,censored=False)
        except TranscriptCapacity as error:
            row=dict(seed=seed,censored=True,reason=str(error),done=False)
        row.update(generated_tokens=sum(r=='assistant' for r in client.roles),
                   raw_transcript_tokens=len(client.tokens),seconds=time.perf_counter()-client.started,
                   attention=args.attention,window=args.window,generation_seed=generation_seed(seed))
        client.save(row)
        receipt.write_text(json.dumps(row,indent=2))
        rows.append(row)
        summary=dict(episodes=len(rows),completed=sum(bool(r.get('done')) for r in rows),
                     censored=sum(r['censored'] for r in rows),
                     completion_fraction_of_started=sum(bool(r.get('done')) for r in rows)/len(rows),
                     episodes_detail=rows)
        (output/f'summary-{args.start_seed}.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(row),flush=True)
        client.stream=None
        client.pinned=None
        torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--attention',choices=['full','ps'],required=True)
    parser.add_argument('--window',type=int,default=1024)
    parser.add_argument('--start-seed',type=int,default=1_000_000)
    parser.add_argument('--count',type=int,default=64)
    parser.add_argument('--transcript-cap',type=int,default=262144)
    parser.add_argument('--temperature',type=float,default=.6)
    parser.add_argument('--top-p',type=float,default=.95)
    parser.add_argument('--top-k',type=int,default=20)
    run(parser.parse_args())


if __name__=='__main__':
    main()
