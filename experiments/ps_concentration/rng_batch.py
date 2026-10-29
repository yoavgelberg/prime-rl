"""Batch native game clients through the established streaming decoder."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
from queue import Empty, Queue
import re
import sys
from threading import Event, local
import time
import types

import torch
from transformers import AutoTokenizer

from .policy import load_model, StreamingQwen
from .rng_stream import StreamingClient, TranscriptCapacity, generation_seed, text_content


class LaneCache:
    def __init__(self, shared, lane, owner):
        self.shared, self.lane, self.owner = shared, lane, owner

    @property
    def keys(self):
        return [k[self.lane:self.lane+1] for k in self.shared.keys]

    @property
    def cache_bytes(self):
        used = min(self.owner.consumed,self.shared.capacity)
        return sum(k.shape[1]*used*k.shape[-1]*k.element_size()*2 for k in self.keys)


class LaneClient(StreamingClient):
    def __init__(self, base, tokenizer, args, requests, index):
        super().__init__(base,tokenizer,args)
        self.requests, self.index = requests,index
        self.queue = deque()
        self.done = False
        self.abort_error = None

    def chat(self,messages):
        if self.abort_error is not None:
            raise self.abort_error
        request = dict(messages=messages,event=Event(),error=None)
        self.pending = request
        self.requests.put((self.index,request))
        request['event'].wait()
        if request['error'] is not None:
            raise request['error']
        return request['response']


class BatchDecoder:
    def __init__(self, clients, requests, args):
        self.clients, self.requests, self.args = clients,requests,args
        self.tokenizer = clients[0].tokenizer
        self.base = clients[0].base
        native_eos = self.base.generation_config.eos_token_id
        self.eos = set(native_eos if isinstance(native_eos,list) else [native_eos])
        self.newline = self.tokenizer.encode('\n',add_special_tokens=False)[0]
        thinking_end=self.tokenizer.encode('</think>',add_special_tokens=False)
        assert len(thinking_end)==1
        self.thinking_end=thinking_end[0]

    def _fail(self,lane,error):
        lane.done = True
        lane.request['error'] = error
        lane.request['event'].set()

    def _append(self,lane,tokens,role,sample_after=False):
        lane._check_capacity(len(tokens))
        start = len(lane.tokens)
        lane.tokens.extend(tokens)
        lane.roles.extend([role]*len(tokens))
        lane.old_logps.extend([None]*len(tokens))
        lane.queue.extend((token,sample_after and j==len(tokens)-1,start+j) for j,token in enumerate(tokens))

    def _request(self,index,request):
        lane = self.clients[index]
        if request is None:
            lane.done = True
            return
        lane.request = request
        new = [dict(role=m['role'],content=text_content(m)) for m in request['messages'][lane.seen:]]
        assert new and all(m['role']=='user' for m in new)
        rendered = self.tokenizer.apply_chat_template([lane.system,*new],tokenize=True,add_generation_prompt=True)
        assert rendered[:len(lane.prefix)]==lane.prefix
        closing = lane.close[1:] if lane.tokens[-1]==lane.close[0] else lane.close
        try:
            lane._check_capacity(len(closing)+len(rendered)-len(lane.prefix))
            self._append(lane,closing,'template')
            self._append(lane,rendered[len(lane.prefix):],'environment_template',True)
        except TranscriptCapacity as error:
            self._fail(lane,error)
            return
        lane.sampled, lane.sample_logps = [],[]
        lane.final_start = None
        lane.turn_start = len(lane.tokens)
        lane.turn_started = time.perf_counter()

    def _emit_many(self,indices,logits):
        active=[]
        for index in indices:
            lane=self.clients[index]
            try:
                lane._check_capacity(1)
            except TranscriptCapacity as error:
                self._fail(lane,error)
                continue
            active.append(index)
        if not active:
            return
        raw=logits[active].float()
        values,choices=(raw/self.args.temperature).topk(self.args.top_k,dim=-1)
        probs=values.softmax(-1)
        probs.masked_fill_(probs.cumsum(-1)-probs>=self.args.top_p,0.)
        probs/=probs.sum(-1,keepdim=True)
        selected=torch.stack([torch.multinomial(probs[row],1,generator=self.clients[index].rng)[0]
                              for row,index in enumerate(active)])
        tokens=choices.gather(1,selected[:,None])[:,0]
        logps=raw.gather(1,tokens[:,None])[:,0]-raw.logsumexp(-1)
        pairs=torch.stack((tokens.float(),logps),dim=-1).tolist()
        for index,(token,logp) in zip(active,pairs,strict=True):
            self._emit(index,int(token),logp)

    def _emit(self,index,token,logp):
        lane = self.clients[index]
        lane.sampled.append(token)
        lane.sample_logps.append(logp)
        if token==self.thinking_end:
            lane.final_start=len(lane.sampled)
        stop = 'eos' if token in self.eos else None
        if stop is None and token==self.newline and lane.final_start is not None:
            text = self.tokenizer.decode(lane.sampled[lane.final_start:],skip_special_tokens=True)
            if re.search(r'(?:^|\n)Action:\s*[a-z][A-Z][ \t]*\n',text):
                stop = 'action_line'
        position = len(lane.tokens)
        self._append(lane,[token],'assistant',stop is None)
        lane.old_logps[position] = logp
        if stop is None:
            return
        text = self.tokenizer.decode(lane.sampled,skip_special_tokens=True)
        reasoning, final = text.rsplit('</think>',1) if '</think>' in text else (text,'')
        turn = dict(start=lane.turn_start,end=len(lane.tokens),sampled_ids=lane.sampled,
                    old_logps=lane.sample_logps,stop_condition=stop,
                    seconds=time.perf_counter()-lane.turn_started,reasoning=reasoning,content=final.strip())
        lane.turns.append(turn)
        lane.seen = len(lane.request['messages'])+1
        progress=dict(seed=lane.seed,attention=self.args.attention,turns=len(lane.turns),
                      raw_transcript_tokens=len(lane.tokens),generated_tokens=sum(r=='assistant' for r in lane.roles),
                      seconds=time.perf_counter()-lane.started,batch_size=len(self.clients),latest_action=final.strip())
        (lane.output/f'progress-{lane.seed}-batch.json').write_text(json.dumps(progress,indent=2))
        with gzip.open(lane.output/f'seed-{lane.seed}-turns.jsonl.gz','at') as handle:
            handle.write(json.dumps(turn)+'\n')
        lane.request['response'] = dict(content=final.strip(),reasoning=reasoning)
        lane.request['event'].set()
        print(json.dumps(progress),flush=True)

    @torch.inference_mode()
    def run(self):
        initial = {}
        while len(initial)<len(self.clients):
            index,request=self.requests.get()
            assert request is not None
            initial[index]=request
        rendered=[]
        for index,lane in enumerate(self.clients):
            request=initial[index]
            messages=[dict(role=m['role'],content=text_content(m)) for m in request['messages']]
            lane.system=messages[0]
            lane.prefix=self.tokenizer.apply_chat_template([lane.system],tokenize=True,add_generation_prompt=False)
            assert len(lane.prefix)<=512
            marker='UNIQUE_CONTENT_MARKER_8037'
            close=self.tokenizer.apply_chat_template([dict(role='assistant',content=marker)],tokenize=False,add_generation_prompt=False)
            lane.close=self.tokenizer.encode(close.split(marker,1)[1],add_special_tokens=False)
            tokens=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
            assert tokens[:len(lane.prefix)]==lane.prefix
            assert self.tokenizer.decode(tokens[-8:],skip_special_tokens=False).rstrip().endswith('<think>')
            lane.tokens=tokens.copy()
            lane.roles=['prefix']*len(lane.prefix)+['environment_template']*(len(tokens)-len(lane.prefix))
            lane.old_logps=[None]*len(tokens)
            lane.request=request
            lane.sampled,lane.sample_logps=[],[]
            lane.final_start=None
            lane.turn_start=len(tokens)
            lane.consumed=len(tokens)
            lane.turn_started=time.perf_counter()
            rendered.append(tokens)
        assert all(tokens==rendered[0] for tokens in rendered)
        shared=StreamingQwen(self.base,len(self.clients),len(self.clients[0].prefix),self.args.window,
                             self.args.attention,self.args.transcript_cap)
        for index,lane in enumerate(self.clients):
            lane.stream=LaneCache(shared,index,lane)
        logits=shared.prefill(torch.tensor(rendered,device=self.base.device))
        pinned=[k[:,:,:shared.prefix].clone() for k in shared.keys]
        self._emit_many(list(range(len(self.clients))),logits)
        ticks=len(rendered[0])
        while not all(lane.done for lane in self.clients):
            while any(not lane.done and not lane.queue for lane in self.clients):
                self._request(*self.requests.get())
            while True:
                try:
                    item=self.requests.get_nowait()
                except Empty:
                    break
                self._request(*item)
            if all(lane.done for lane in self.clients):
                break
            inputs,requesting=[],[]
            for index,lane in enumerate(self.clients):
                if lane.done:
                    inputs.append(self.tokenizer.eos_token_id)
                    continue
                token,sample_after,position=lane.queue.popleft()
                assert position==ticks,(index,position,ticks)
                lane.consumed=position+1
                inputs.append(token)
                if sample_after:
                    requesting.append(index)
            logits=shared.step(torch.tensor(inputs,device=self.base.device))
            ticks+=1
            self._emit_many(requesting,logits)
        for keys,old in zip(shared.keys,pinned,strict=True):
            assert torch.equal(keys[:,:,:shared.prefix],old)
        return ticks


def run(args):
    source=Path(args.source)
    sys.path.insert(0,str(source/'1_matching_pairs_new'))
    state=local()
    shim=types.ModuleType('model_presets')
    shim.make_client=lambda *a,**kw:state.client
    shim.parse_grid_size=lambda value:tuple(map(int,value.split('x')))
    sys.modules['model_presets']=shim
    from modes.single_normal import run_one_game
    from common import prompts
    prompts.OUTPUT_FORMAT_NO_COT='''Output format:
After thinking, end your final response with exactly one action line:
Action: <coordinate, e.g. cD>
'''
    base=load_model(args.model)
    requested_batch_size=args.batch_size
    if args.attention=='full':
        kv_bytes=sum(4*layer.self_attn.k_proj.weight.shape[0]*args.transcript_cap for layer in base.model.layers)
        cast_bytes=sum(p.numel()*2 for p in base.parameters())
        free_bytes,_=torch.cuda.mem_get_info()
        slots=(free_bytes-cast_bytes-2*1024**3)//kv_bytes
        assert slots>=1,dict(free_bytes=free_bytes,kv_bytes=kv_bytes,cast_bytes=cast_bytes)
        args.batch_size=min(args.batch_size,int(slots))
        print(json.dumps(dict(event='memory_admission',requested_batch_size=requested_batch_size,
                              admitted_batch_size=args.batch_size,free_bytes=free_bytes,
                              kv_bytes_per_trajectory=kv_bytes)),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    metadata=dict(vars(args),requested_batch_size=requested_batch_size,
                  model_revision=Path(args.model).name,thinking_retained=True,
                  system_prompt=prompts.build_single_system(cot_enabled=False),
                  upstream=json.loads((source/'SOURCE.json').read_text()),
                  arithmetic='bf16_products_attention_fp32_residual_and_projection_accumulation',
                  gpu=torch.cuda.get_device_name(),scope='Batched base qualification; no RL/SFT')
    (output/f'config-{args.start_seed}-batch{args.batch_size}-shard{args.shard_count}-{args.shard_index}.json').write_text(json.dumps(metadata,indent=2))
    seeds=[s for s in range(args.start_seed,args.start_seed+args.count)
           if (s-args.start_seed)%args.shard_count==args.shard_index and not (output/f'seed-{s}-result.json').exists()]
    for offset in range(0,len(seeds),args.batch_size):
        boards=seeds[offset:offset+args.batch_size]
        requests=Queue()
        clients=[LaneClient(base,tokenizer,args,requests,i) for i in range(len(boards))]
        for client,seed in zip(clients,boards,strict=True):
            client.reset(seed,output)
        def play(index):
            client=clients[index];seed=boards[index];state.client=client
            try:
                try:
                    result=run_one_game(model='local',rows=4,cols=4,num_cards=None,seed=seed,render='text',
                        theme=None,assets_dir=str(source/'1_matching_pairs_new/assets'),cell_size=64,
                        out_dir=output/f'seed-{seed}',max_responses=args.max_responses,max_retries=2,cot_enabled=False)
                    assert result.get('error') is None,result
                    assert result['response_count']<=args.max_responses
                    row=dict(seed=seed,**result,censored=False,turn_cap_failure=not result['done'],
                             terminal_outcome_reward=1. if result['done'] else -1.)
                except TranscriptCapacity as error:
                    row=dict(seed=seed,censored=True,reason=str(error),done=False)
                row.update(generated_tokens=sum(r=='assistant' for r in client.roles),raw_transcript_tokens=len(client.tokens),
                           seconds=time.perf_counter()-client.started,attention=args.attention,window=args.window,
                           generation_seed=generation_seed(seed),max_responses=args.max_responses,
                           batch_size=len(boards),peak_memory_scope='shared GPU worker')
                requests.put((index,None))
                client.save(row)
                (output/f'seed-{seed}-result.json').write_text(json.dumps(row,indent=2))
                progress_path=output/f'progress-{seed}-batch.json'
                if progress_path.exists():
                    progress_path.unlink()
                print(json.dumps(row),flush=True)
                return row
            finally:
                requests.put((index,None))
        started=time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(boards)) as pool:
            futures=[pool.submit(play,index) for index in range(len(boards))]
            try:
                ticks=BatchDecoder(clients,requests,args).run()
            except BaseException:
                for client in clients:
                    error=RuntimeError('Batch decoder aborted; original error is raised by the coordinator')
                    client.abort_error=error
                    if hasattr(client,'pending'):
                        client.pending['error']=error
                        client.pending['event'].set()
                raise
            rows=[future.result() for future in futures]
        summary=dict(seeds=boards,batch_size=len(boards),seconds=time.perf_counter()-started,ticks=ticks,
                     generated_tokens=sum(r['generated_tokens'] for r in rows),
                     completed=sum(bool(r.get('done')) for r in rows),censored=sum(r['censored'] for r in rows))
        (output/f'batch-{boards[0]}-result.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(dict(event='batch_complete',**summary)),flush=True)
        for client in clients:
            client.stream=None
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
    parser.add_argument('--max-responses',type=int,default=96)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--shard-count',type=int,default=1)
    parser.add_argument('--transcript-cap',type=int,default=262144)
    parser.add_argument('--temperature',type=float,default=.6)
    parser.add_argument('--top-p',type=float,default=.95)
    parser.add_argument('--top-k',type=int,default=20)
    run(parser.parse_args())


if __name__=='__main__':
    main()
