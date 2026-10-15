#!/usr/bin/env python3
"""Isolated, same-attention-kernel Q/K RMSNorm fusion experiment.

The primary comparison is native separate RMSNorm + fixed-launch Flash vs the
SAME Flash kernel with RMSNorm in its prologue. All timing uses CUDA graphs.
Neither this isolated kernel nor synthetic config-shaped tensors represent a
complete model, a RoPE-aware attention module, or an E2E run.
"""
from __future__ import annotations
import argparse, gc, gzip, hashlib, json, math, os, platform, random, statistics, time, subprocess, threading
from pathlib import Path
import torch
import torch.nn.functional as F
import torch._dynamo as dynamo
import triton
import triton.language as tl

MODELS = {
 'Qwen3-0.6B': (16, 8, 128),
 'Qwen3-1.7B': (32, 8, 128),
 'Qwen3-4B': (32, 8, 128),
 'Qwen3-8B': (32, 8, 128),
 'Qwen3-32B': (64, 8, 128),
 'Qwen3-30B-A3B': (32, 4, 128),
 'Gemma-3-4B': (8, 4, 256),
 'Gemma-3-12B': (16, 8, 256),
 'Gemma-3-27B': (32, 16, 128),
}
EPS = 1e-6
ATOL = .025
RTOL = .025
RL2 = .025
CAUSAL_MODE = False

@triton.jit
def _norm(X, G, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
 row = tl.program_id(0)
 j = tl.arange(0, BLOCK)
 x = tl.load(X + row*N + j, j < N, 0).to(tl.float32)
 g = tl.load(G+j, j < N, 0).to(tl.float32)
 r = tl.rsqrt(tl.sum(x*x, axis=0)/N + EPS)
 tl.store(Y+row*N+j, (x*r*g).to(Y.dtype.element_ty), j < N)

def native_norm(x, g):
 y = torch.empty_like(x)
 _norm[(x.numel()//x.shape[-1],)](x, g, y, x.shape[-1], EPS, triton.next_power_of_2(x.shape[-1]), num_warps=4)
 return y

@triton.jit
def _flash(Q,K,V,GQ,GK,O,S:tl.constexpr,D:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
           SCALE:tl.constexpr,FUSED:tl.constexpr,CAUSAL:tl.constexpr,
           BM:tl.constexpr,BN:tl.constexpr):
 pid=tl.program_id(0); h=tl.program_id(1); kh=h//(HQ//HK)
 mi=pid*BM+tl.arange(0,BM); dj=tl.arange(0,D)
 q=tl.load(Q+h*S*D+mi[:,None]*D+dj[None,:],mi[:,None]<S,0)
 if FUSED:
  qf=q.to(tl.float32)
  qr=tl.rsqrt(tl.sum(qf*qf,axis=1)/D+1.e-6)
  qg=tl.load(GQ+dj).to(tl.float32)
  q=(qf*qr[:,None]*qg[None,:]).to(q.dtype)
 acc=tl.zeros((BM,D),tl.float32); mx=tl.full((BM,),float('-inf'),tl.float32); den=tl.zeros((BM,),tl.float32)
 for start in range(0,S,BN):
  nj=start+tl.arange(0,BN)
  k=tl.load(K+kh*S*D+nj[:,None]*D+dj[None,:],nj[:,None]<S,0)
  if FUSED:
   kf=k.to(tl.float32)
   kr=tl.rsqrt(tl.sum(kf*kf,axis=1)/D+1.e-6)
   kg=tl.load(GK+dj).to(tl.float32)
   k=(kf*kr[:,None]*kg[None,:]).to(k.dtype)
  s=tl.dot(q,tl.trans(k))*SCALE
  s=tl.where(nj[None,:]<S,s,float('-inf'))
  if CAUSAL:
   s=tl.where(mi[:,None]>=nj[None,:],s,float('-inf'))
  mn=tl.maximum(mx,tl.max(s,axis=1)); alpha=tl.exp(mx-mn); p=tl.exp(s-mn[:,None])
  den=den*alpha+tl.sum(p,axis=1); acc=acc*alpha[:,None]
  v=tl.load(V+kh*S*D+nj[:,None]*D+dj[None,:],nj[:,None]<S,0)
  acc+=tl.dot(p.to(v.dtype),v); mx=mn
 out=acc/den[:,None]
 tl.store(O+h*S*D+mi[:,None]*D+dj[None,:],out.to(O.dtype.element_ty),mi[:,None]<S)

def flash(q,k,v,gq=None,gk=None,fused=False):
 # Fixed launch and identical attention algorithm for both primary arms.
 assert q.shape[0] == 1
 B,H,S,D=q.shape; HK=k.shape[1]
 o=torch.empty_like(q)
 _flash[(triton.cdiv(S,32),H)](q,k,v,gq if gq is not None else q,gk if gk is not None else k,o,S,D,H,HK,D**-.5,fused,CAUSAL_MODE,32,32,num_warps=4,num_stages=1)
 return o

def norm_torch(x,g):
 xf=x.float(); r=torch.rsqrt(xf.square().mean(-1,keepdim=True)+EPS)
 return (xf*r*g.float()).to(x.dtype)

def expand(x,h):
 return x if x.shape[1]==h else x.repeat_interleave(h//x.shape[1],dim=1)

def attn_vjp(q,k,v,dy):
 """Analytical attention VJP, native torch ops, with GQA reduction.

 This is the executed derivative of the candidate operation, not an autograd
 invocation of the reference. The float softmax is recomputed explicitly.
 """
 h=q.shape[1]; hk=k.shape[1]; groups=h//hk
 qq=q.float(); kk=expand(k,h).float(); vv=expand(v,h).float(); ddy=dy.float()
 p=torch.softmax((qq@kk.transpose(-1,-2))*(q.shape[-1]**-.5),dim=-1)
 dv=p.transpose(-1,-2)@ddy; dp=ddy@vv.transpose(-1,-2)
 ds=p*(dp-(dp*p).sum(-1,keepdim=True)); ds=ds*(q.shape[-1]**-.5)
 dq=ds@kk; dk=ds.transpose(-1,-2)@qq
 if groups>1:
  b,_,s,d=dk.shape
  dk=dk.reshape(b,hk,groups,s,d).sum(2); dv=dv.reshape(b,hk,groups,s,d).sum(2)
 return dq.to(q.dtype),dk.to(k.dtype),dv.to(v.dtype)

def norm_vjp(x,g,dy):
 xf=x.float(); r=torch.rsqrt(xf.square().mean(-1,keepdim=True)+EPS)
 u=xf*r; dz=dy.float()*g.float()
 dx=r*(dz-u*(dz*u).mean(-1,keepdim=True))
 dg=(dy.float()*u).sum(tuple(range(x.ndim-1)))
 return dx.to(x.dtype),dg.to(g.dtype)

class Norm(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,g):
  ctx.save_for_backward(x,g); return native_norm(x,g)
 @staticmethod
 def backward(ctx,dy):
  return norm_vjp(*ctx.saved_tensors,dy)

class PlainFlash(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v):
  ctx.save_for_backward(q,k,v); return flash(q,k,v,fused=False)
 @staticmethod
 def backward(ctx,dy):
  return attn_vjp(*ctx.saved_tensors,dy)

class FusedFlash(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,gq,gk):
  ctx.save_for_backward(q,k,v,gq,gk); return flash(q,k,v,gq,gk,fused=True)
 @staticmethod
 def backward(ctx,dy):
  q,k,v,gq,gk=ctx.saved_tensors
  qn=norm_torch(q,gq); kn=norm_torch(k,gk)
  dqn,dkn,dv=attn_vjp(qn,kn,v,dy)
  dq,dgq=norm_vjp(q,gq,dqn); dk,dgk=norm_vjp(k,gk,dkn)
  return dq,dk,dv,dgq,dgk

def baseline(q,k,v,gq,gk):
 return PlainFlash.apply(Norm.apply(q,gq),Norm.apply(k,gk),v)

def candidate(q,k,v,gq,gk):
 return FusedFlash.apply(q,k,v,gq,gk)

def external(q,k,v,gq,gk):
 qn=norm_torch(q,gq); kn=norm_torch(k,gk)
 return F.scaled_dot_product_attention(qn,expand(kn,q.shape[1]),expand(v,q.shape[1]),is_causal=CAUSAL_MODE)

def oracle(q,k,v,gq,gk):
 qn=norm_torch(q,gq); kn=norm_torch(k,gk)
 scores=(qn@expand(kn,q.shape[1]).transpose(-1,-2))*(q.shape[-1]**-.5)
 if CAUSAL_MODE:
  s=scores.shape[-1]; mask=torch.ones((s,s),device=scores.device,dtype=torch.bool).triu(1); scores=scores.masked_fill(mask,float('-inf'))
 p=torch.softmax(scores,dim=-1)
 return p@expand(v,q.shape[1])

def metric(x,y):
 a=x.detach().float(); b=y.detach().float(); diff=a-b
 l2=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(b).clamp_min(1e-12))
 close=bool(torch.allclose(a,b,atol=ATOL,rtol=RTOL))
 return {'allclose':close, 'relative_l2':l2,'max_abs':float(diff.abs().max()),'rms_abs':float(diff.square().mean().sqrt()),'pass':close and l2<=RL2}

def inputs(cfg,s,seed,grad=False,dtype=torch.bfloat16):
 torch.manual_seed(seed); h,hk,d=cfg
 shape=lambda heads:(1,heads,s,d)
 xs=[torch.randn(shape(h),device='cuda',dtype=dtype)*.5,torch.randn(shape(hk),device='cuda',dtype=dtype)*.5,torch.randn(shape(hk),device='cuda',dtype=dtype)*.5,1+torch.randn((d,),device='cuda',dtype=dtype)*.05,1+torch.randn((d,),device='cuda',dtype=dtype)*.05]
 xs=[x.contiguous().requires_grad_(grad) for x in xs]
 dy=torch.randn(shape(h),device='cuda',dtype=dtype)
 return xs,dy

def correctness(cfg,s):
 records=[]
 for seed in (1701,2803,3907):
  # Correctness is checked in FP32. Timed graphs below intentionally use BF16.
  xs,dy=inputs(cfg,s,seed,True,dtype=torch.float32)
  refxs=[x.detach().float().requires_grad_() for x in xs]
  ref=oracle(*refxs); refgr=torch.autograd.grad(ref,refxs,dy.float())
  b=baseline(*xs); bg=torch.autograd.grad(b,xs,dy)
  c=candidate(*xs); cg=torch.autograd.grad(c,xs,dy)
  records.append({'seed':seed,'forward_baseline_fp32':metric(b,ref),'forward_candidate_fp32':metric(c,ref),'forward_candidate_baseline':metric(c,b),'grad_baseline_fp32':[metric(a,z) for a,z in zip(bg,refgr)],'grad_candidate_fp32':[metric(a,z) for a,z in zip(cg,refgr)],'grad_candidate_baseline':[metric(a,z) for a,z in zip(cg,bg)]})
 def passes(r):
  return all(r[k]['pass'] for k in ['forward_baseline_fp32','forward_candidate_fp32','forward_candidate_baseline']) and all(x['pass'] for k in ['grad_baseline_fp32','grad_candidate_fp32','grad_candidate_baseline'] for x in r[k])
 return {'seeds':records,'pass':all(passes(r) for r in records)}

def capture(fn, cfg,s,seed,training=False):
 xs,dy=inputs(cfg,s,seed,training); stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
 def body():
  y=fn(*xs)
  return torch.autograd.grad(y,xs,dy) if training else y
 with torch.cuda.stream(stream):
  for _ in range(20): out=body()
 torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph): out=body()
 return graph,(xs,dy,out)

def stats(xs):
 if not xs:
  raise ValueError('timing sample collection is empty')
 vals=[float(x) for x in xs]
 if not all(math.isfinite(x) and x >= 0.0 for x in vals):
  raise ValueError(f'timing samples must be finite and non-negative: {vals!r}')
 mean=statistics.fmean(vals)
 # Report the uncertainty required by the publication protocol.  The stored
 # samples are the event-level observations; ``two_sem_ms`` is 2*SEM over
 # those observations (and is retained alongside the mean rather than being
 # reconstructed from rounded medians).
 sem=statistics.stdev(vals)/math.sqrt(len(vals)) if len(vals)>1 else 0.0
 return {'n':len(vals),'mean_ms':mean,'sem_ms':sem,'two_sem_ms':2.0*sem,
         'p50_ms':statistics.median(vals),
         'p95_ms':sorted(vals)[math.ceil(.95*len(vals))-1],'samples_ms':vals}

def measure(cfg,s,rounds,samples,fn0=baseline,fn1=candidate,training=False,seed_offset=0):
 # Each round captures fresh, separate CUDA graphs after 20 warmup calls.
 # The arm order is independently randomized per paired sample.
 if rounds < 1 or samples < 2:
  raise ValueError('paired timing requires rounds >= 1 and samples >= 2')
 rr=[]; rng=random.Random(98431+s+int(training)+seed_offset)
 for rnd in range(rounds):
  a,ka=capture(fn0,cfg,s,1701,training); b,kb=capture(fn1,cfg,s,1701,training)
  for _ in range(20): a.replay(); b.replay()
  torch.cuda.synchronize(); aa=[]; bb=[]; order_log=[]
  for _ in range(samples):
   order=[(a,aa),(b,bb)]
   if rng.random()<.5:
    order.reverse(); order_log.append('candidate_first')
   else:
    order_log.append('baseline_first')
   for graph,vals in order:
    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
    start.record(); graph.replay(); end.record(); end.synchronize(); vals.append(float(start.elapsed_time(end)))
  ds=[x-y for x,y in zip(aa,bb)]
  rr.append({'round':rnd,'baseline':stats(aa),'candidate':stats(bb),'arm_order':order_log,
             'ratio':statistics.median(aa)/statistics.median(bb),
             'paired_delta_mean_ms':statistics.fmean(ds),
             'paired_delta_ci95_ms':1.96*statistics.stdev(ds)/math.sqrt(len(ds))})
  del a,b,ka,kb; gc.collect(); torch.cuda.empty_cache()
 ratios=[x['ratio'] for x in rr]; means=[x['paired_delta_mean_ms'] for x in rr]
 return {'mode':'cuda_graph_replay_paired_interleaved','paired':True,
         'arm_order':'randomized_per_pair_sample',
         'fresh_graph_captures':rounds,'warmups_per_capture':20,
         'samples_per_round':samples,'rounds':rr,
         'ratio_median_round':statistics.median(ratios),
         'ratio_range_round':[min(ratios),max(ratios)],
         'delta_mean_across_rounds_ms':statistics.fmean(means),
         'delta_ci95_across_rounds_ms':1.96*statistics.stdev(means)/math.sqrt(len(means)) if len(means)>1 else None,
         'seed_offset':seed_offset}

def paired_arm_record(paired, arm, label):
 """Project one paired arm into the regular timing-record shape."""
 if arm not in ('baseline','candidate'):
  raise ValueError(f'unknown paired arm: {arm}')
 rounds=[{'round':r['round'],'stats':r[arm]} for r in paired['rounds']]
 med=[r['stats']['p50_ms'] for r in rounds]
 return {'label':label,'mode':'cuda_graph_replay_paired_interleaved',
         'paired':True,'arm_order':paired.get('arm_order'),
         'fresh_graph_captures':paired['fresh_graph_captures'],
         'warmup_per_round':paired['warmups_per_capture'],
         'samples_per_round':paired['samples_per_round'],'rounds':rounds,
         'median_ms':statistics.median(med),
         'mean_round_median_ms':statistics.fmean(med),
         'round_medians_ms':med}

def _finite_number(value):
 try:
  return isinstance(value,(int,float)) and math.isfinite(float(value))
 except (OverflowError,TypeError,ValueError):
  return False

def validate_timing_record(record, rounds, samples, paired=False):
 """Return ``(ok, reasons)`` for a timing record used by the gate."""
 reasons=[]
 if not isinstance(record,dict):
  return False,['timing record is not an object']
 if record.get('mode') not in ('cuda_graph_replay','cuda_graph_replay_paired_interleaved'):
  reasons.append(f"unexpected timing mode {record.get('mode')!r}")
 if paired and not (record.get('paired') is True and record.get('mode') == 'cuda_graph_replay_paired_interleaved'):
  reasons.append('required paired/interleaved timing is absent')
 if paired and record.get('arm_order') != 'randomized_per_pair_sample':
  reasons.append('timing arm order was not recorded as randomized per pair')
 warmup_key='warmups_per_capture' if 'warmups_per_capture' in record else 'warmup_per_round'
 if record.get(warmup_key) != 20:
  reasons.append('timing warmup count does not match required 20 calls')
 rs=record.get('rounds')
 if record.get('fresh_graph_captures') != rounds or record.get('samples_per_round') != samples:
  reasons.append('timing round/sample counts do not match requested protocol')
 if not isinstance(rs,list) or len(rs) != rounds:
  reasons.append('timing round count is incomplete')
 else:
  for idx,entry in enumerate(rs):
   if not isinstance(entry,dict):
    reasons.append(f'timing round {idx} is not an object')
    continue
   if paired and 'baseline' in entry:
    order_log=entry.get('arm_order')
    if not isinstance(order_log,list) or len(order_log) != samples or not all(x in ('baseline_first','candidate_first') for x in order_log):
     reasons.append(f'timing round {idx} is missing randomized arm-order evidence')
   stat=entry.get('stats') if 'stats' in entry else None
   if paired:
    stat_candidates=[stat] if stat is not None else [entry.get('baseline'),entry.get('candidate')]
   else:
    stat_candidates=[stat]
   for candidate_stat in stat_candidates:
    if not isinstance(candidate_stat,dict) or candidate_stat.get('n') != samples:
     reasons.append(f'timing round {idx} has an incomplete sample count')
     continue
    vals=candidate_stat.get('samples_ms')
    if not isinstance(vals,list) or len(vals) != samples or not all(_finite_number(x) and float(x) >= 0.0 for x in vals):
     reasons.append(f'timing round {idx} has non-finite or missing samples')
    for key in ('mean_ms','sem_ms','two_sem_ms','p50_ms','p95_ms'):
     if not _finite_number(candidate_stat.get(key)):
      reasons.append(f'timing round {idx} is missing finite {key}')
 if not _finite_number(record.get('median_ms',record.get('ratio_median_round',float('nan')))):
  reasons.append('timing aggregate is not finite')
 return not reasons,reasons

def publication_gate(row, rounds, samples):
 """Evaluate every required evidence source before assigning tested_pass."""
 if not isinstance(row,dict):
  return {'pass':False,'reasons':['row is not an object']}
 reasons=[]
 if row.get('errors'):
  reasons.append('one or more execution errors were recorded')
 correctness=row.get('correctness')
 if not isinstance(correctness,dict) or correctness.get('pass') is not True:
  reasons.append('correctness did not pass')
 bf16=row.get('bf16_correctness')
 if not isinstance(bf16,dict) or bf16.get('pass') is not True:
  reasons.append('timed BF16 correctness did not pass')
 compiled_corr=row.get('torchinductor_correctness')
 if not isinstance(compiled_corr,dict) or compiled_corr.get('pass') is not True:
  reasons.append('TorchInductor compiled-arm correctness did not pass')
 if row.get('torchdynamo_reset_before_compile') is not True:
  reasons.append('TorchDynamo was not reset before this configuration')
 ex=row.get('gpu_exclusivity')
 if not isinstance(ex,dict) or ex.get('before',{}).get('exclusive') is not True or ex.get('after',{}).get('exclusive') is not True:
  reasons.append('selected GPU was not proven exclusive at row boundaries')
 elif not ex.get('observations') or not all(x.get('exclusive') is True for x in ex['observations']):
  reasons.append('continuous GPU exclusivity monitor failed or observed another compute PID')
 timing=row.get('timing')
 if not isinstance(timing,dict):
  reasons.append('timing record is missing')
 else:
  for key in ('t_B_norm','t_G_attention_only','t_F_unfused','t_F_fused','vendor_sdpa','torchinductor_compiled'):
   ok,why=validate_timing_record(timing.get(key),rounds,samples,paired=key in ('t_F_unfused','t_F_fused'))
   if not ok: reasons.extend(f'{key}: {msg}' for msg in why)
  pair=timing.get('t_F_pair')
  ok,why=validate_timing_record(pair,rounds,samples,paired=True)
  if not ok: reasons.extend(f't_F_pair: {msg}' for msg in why)
  # Any initial speedup above the 1.01 gate must have an independent fresh
  # paired rerun.  The rerun itself is checked with the same strict protocol.
  initial_ratio=pair.get('ratio_median_round') if isinstance(pair,dict) else None
  if _finite_number(initial_ratio) and float(initial_ratio)>1.01:
   retry=timing.get('t_F_retest')
   if not isinstance(retry,dict):
    reasons.append('initial ratio >1.01 but required independent retest is missing')
   else:
    ok2,why2=validate_timing_record(retry,rounds,samples,paired=True)
    if not ok2: reasons.extend(f't_F_retest: {msg}' for msg in why2)
    if retry.get('seed_offset')==pair.get('seed_offset'):
     reasons.append('t_F_retest reused the initial timing seed')
 if row.get('torchinductor_status') != 'compiled':
  reasons.append('TorchInductor did not compile')
 for key in ('torchinductor_runtime_error','torchinductor_trace_error'):
  if key in row: reasons.append(f'{key} was recorded')
 traces=row.get('traces')
 if not isinstance(traces,list) or not traces:
  reasons.append('required traces are missing')
 else:
  expected_traces=5 if row.get('torchinductor_status') == 'compiled' else 4
  if len(traces) != expected_traces:
   reasons.append(f'required trace count is {expected_traces}, got {len(traces)}')
  try:
   if len(set(traces)) != len(traces):
    reasons.append('trace paths are duplicated')
  except TypeError:
   reasons.append('trace paths are not hashable strings')
  for path in traces:
   try:
    if not Path(path).is_file() or Path(path).stat().st_size <= 0:
     reasons.append(f'trace is missing or empty: {path}')
   except (OSError,TypeError):
    reasons.append(f'trace path is invalid: {path!r}')
 return {'pass':not reasons,'reasons':reasons}

def measure_single(cfg,s,rounds,samples,fn,label):
 # Each round gets a fresh CUDA-graph capture, 20 warmups, and ``samples``.
 rr=[]
 for rnd in range(rounds):
  graph,keep=capture(fn,cfg,s,1701,False)
  for _ in range(20): graph.replay()
  torch.cuda.synchronize(); vals=[]
  for _ in range(samples):
   st=torch.cuda.Event(enable_timing=True); en=torch.cuda.Event(enable_timing=True)
   st.record(); graph.replay(); en.record(); en.synchronize(); vals.append(float(st.elapsed_time(en)))
  rr.append({'round':rnd,'stats':stats(vals)})
  del graph,keep; gc.collect(); torch.cuda.empty_cache()
 med=[x['stats']['p50_ms'] for x in rr]
 return {'label':label,'mode':'cuda_graph_replay','fresh_graph_captures':rounds,
         'warmup_per_round':20,'samples_per_round':samples,'rounds':rr,
         'median_ms':statistics.median(med),'mean_round_median_ms':statistics.fmean(med),
         'round_medians_ms':med}

def raw_attention(q,k,v,gq,gk):
 return flash(q,k,v,fused=False)

def norm_pair(q,k,v,gq,gk):
 return native_norm(q,gq),native_norm(k,gk)

def forward_check(cfg,s):
 records=[]
 for seed in (1701,2803,3907):
  xs,_=inputs(cfg,s,seed,False,dtype=torch.float32)
  ref=oracle(*xs)
  b=baseline(*xs); c=candidate(*xs); v=external(*xs)
  records.append({'seed':seed,'baseline':metric(b,ref),'candidate':metric(c,ref),
                  'candidate_vs_baseline':metric(c,b),'vendor_sdpa':metric(v,ref)})
 return {'seeds':records,'pass':all(r['baseline']['pass'] and r['candidate']['pass'] and
                                    r['candidate_vs_baseline']['pass'] and r['vendor_sdpa']['pass']
                                    for r in records)}

def forward_check_bf16(cfg,s):
 """Check the actual timed BF16 tensors against an FP32 oracle."""
 records=[]
 for seed in (1701,2803,3907):
  xs,_=inputs(cfg,s,seed,False,dtype=torch.bfloat16)
  ref=oracle(*[x.float() for x in xs])
  b=baseline(*xs); c=candidate(*xs); v=external(*xs)
  records.append({'seed':seed,'baseline':metric(b,ref),'candidate':metric(c,ref),
                  'candidate_vs_baseline':metric(c,b),'vendor_sdpa':metric(v,ref)})
 return {'dtype':'bfloat16','oracle_dtype':'float32','seeds':records,
         'pass':all(r['baseline']['pass'] and r['candidate']['pass'] and
                    r['candidate_vs_baseline']['pass'] and r['vendor_sdpa']['pass']
                    for r in records)}

def compiled_forward_check(ext,cfg,s):
 """Check the compiled arm on BF16 inputs against the FP32 oracle."""
 records=[]
 for seed in (1701,2803,3907):
  xs,_=inputs(cfg,s,seed,False,dtype=torch.bfloat16)
  ref=oracle(*[x.float() for x in xs])
  y=ext(*xs)
  records.append({'seed':seed,'compiled_vs_fp32':metric(y,ref),'pass':metric(y,ref)['pass']})
 return {'dtype':'bfloat16','oracle_dtype':'float32','seeds':records,
         'pass':all(x['pass'] for x in records)}

def gpu_exclusivity_snapshot():
 """Record selected physical GPU and foreign compute PIDs at a row boundary."""
 visible=os.getenv('CUDA_VISIBLE_DEVICES','').strip()
 try: physical=[int(x.strip()) for x in visible.split(',') if x.strip()]
 except ValueError: physical=[]
 try:
  q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],
                   check=True,capture_output=True,text=True,timeout=10)
  uuids={int(line.split(',')[0].strip()):line.split(',')[1].strip() for line in q.stdout.splitlines() if ',' in line}
  app=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
                     check=True,capture_output=True,text=True,timeout=10)
  own=os.getpid(); foreign=[]; seen=[]
  selected_uuids={uuids.get(i) for i in physical}
  for line in app.stdout.splitlines():
   if ',' not in line: continue
   u,p=line.split(',',1); u=u.strip(); p=int(p.strip())
   if u in selected_uuids:
    seen.append({'gpu_uuid':u,'pid':p,'self':p==own})
    if p!=own: foreign.append({'gpu_uuid':u,'pid':p})
  return {'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'physical_gpus':physical,'pid':own,'apps':seen,'foreign_apps':foreign,
          'exclusive':bool(physical) and not foreign}
 except Exception as e:
  return {'physical_gpus':physical,'pid':os.getpid(),'apps':[],'foreign_apps':[],
          'exclusive':False,'error':repr(e)}

class ExclusiveMonitor:
 def __init__(self):
  self.stop=threading.Event(); self.records=[]
  self.thread=threading.Thread(target=self.loop,daemon=True)
 def loop(self):
  while not self.stop.is_set():
   self.records.append(gpu_exclusivity_snapshot())
   self.stop.wait(1.0)
 def start(self): self.thread.start()
 def finish(self):
  self.stop.set(); self.thread.join(timeout=25)
  return self.records

def trace(fn,cfg,s,path,training=False):
 from fusion_harness import capture_gpu_trace
 xs,dy=inputs(cfg,s,1701,training)
 def call():
  y=fn(*xs)
  if training: torch.autograd.grad(y,xs,dy)
 capture_gpu_trace(call,path,'candidate.forward_backward' if training else 'candidate.forward')

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); ap.add_argument('--models',default=','.join(MODELS)); ap.add_argument('--sizes',default='128,256,384,512,768,1024,1536,2048'); ap.add_argument('--causal-modes',default='false,true'); ap.add_argument('--rounds',type=int,default=5); ap.add_argument('--samples',type=int,default=100); a=ap.parse_args()
 torch.backends.cuda.matmul.allow_tf32=False; torch.set_float32_matmul_precision('highest')
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
 global CAUSAL_MODE
 rec={'schema':'fusionagent-e1-qknorm-forward-v1','scope':'isolated Q/K RMSNorm into attention prologue; B=1; no projection/RoPE/residual/checkpoint','candidate':'native Triton RMSNorm prologue fixed Flash attention','baseline':'same attention kernel with separate native Triton RMSNorm','additional_arms':['attention_without_norm','TorchInductor compiled SDPA','vendor SDPA/cuDNN'],'claim_boundary':'forward-only publication protocol; model geometry and FP32 oracle consistency are recorded; no fused backward/update claim','source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'environment':{'host':platform.node(),'gpu':torch.cuda.get_device_name(),'visible_gpu':os.getenv('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'triton':triton.__version__,'cuda':torch.version.cuda},'protocol':{'S':list(map(int,a.sizes.split(','))),'causal_modes':[x.strip()=='true' for x in a.causal_modes.split(',')],'warmups':20,'warmup':20,'samples':a.samples,'interleaved_rounds':a.rounds,'rounds':a.rounds,'cuda_graph':True,'interleaved':True,'timing':'paired/interleaved randomized baseline vs candidate; independent captures for additional arms','correctness':'FP32 oracle, baseline/candidate/vendor forward outputs'},'correctness_thresholds':{'atol':ATOL,'rtol':RTOL,'relative_l2':RL2},'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'models':{},'status':'running','publication_eligible':False}
 rec['protocol']['timing_required_arms']=['t_B_norm','t_G_attention_only','t_F_unfused','t_F_fused','t_F_retest_when_ratio_gt_1.01','vendor_sdpa','torchinductor_compiled']
 rec['protocol']['paired_timing']='t_F_unfused/t_F_fused; arm_order=randomized_per_pair_sample'
 rec['protocol']['interleaved']=True
 rec['protocol']['timed_dtype']='bfloat16'
 rec['protocol']['correctness']='FP32 and timed BF16 arms against FP32 oracle; compiled BF16 arm checked; no backward/update claim'
 rec['protocol']['torchdynamo_reset']='torch._dynamo.reset() immediately before each shape compile'
 rec['protocol']['gpu_exclusivity']='nvidia-smi selected physical GPU sampled at row start/end and every second; any foreign PID invalidates row'
 for causal in [x.strip().lower()=='true' for x in a.causal_modes.split(',')]:
  CAUSAL_MODE=causal
  for name in a.models.split(','):
   cfg=MODELS[name]; rows=[]
   for s in map(int,a.sizes.split(',')):
    print('RUN',name,'causal',causal,'S',s,flush=True)
    row={'S':s,'causal':causal,'timed_dtype':'bfloat16','correctness':{'pass':False},'errors':[]}
    row['gpu_exclusivity']={'poll_seconds':1.0,'before':gpu_exclusivity_snapshot()}
    if not row['gpu_exclusivity']['before']['exclusive']:
     raise RuntimeError('Selected GPU is occupied or exclusivity cannot be checked: '+repr(row['gpu_exclusivity']['before']))
    monitor=ExclusiveMonitor(); monitor.start()
    try:
     row['correctness']=forward_check(cfg,s)
     row['bf16_correctness']=forward_check_bf16(cfg,s)
     print('CORRECTNESS',name,causal,s,row['correctness']['pass'],flush=True)
    except Exception as e:
     row['errors'].append({'stage':'correctness','error':repr(e)})
     print('CORRECTNESS_ERROR',name,causal,s,repr(e),flush=True)
    ext=None; compiled_status='not_run'
    try:
     dynamo.reset(); row['torchdynamo_reset_before_compile']=True
     ext=torch.compile(external,fullgraph=True); compiled_status='compiled'
     row['torchinductor_correctness']=compiled_forward_check(ext,cfg,s)
    except Exception as e:
     row['torchinductor_compile_error']=repr(e); compiled_status='compile_error'
    try:
     if not all(row.get(k,{}).get('pass') is True for k in ('correctness','bf16_correctness','torchinductor_correctness')):
      raise RuntimeError('pre-timing correctness gate failed; timing forbidden')
     paired=measure(cfg,s,a.rounds,a.samples,baseline,candidate)
     # Retest only configurations whose initial paired speedup crosses the
     # pre-registered 1.01 threshold.  This is a fresh capture/sample set,
     # with a distinct RNG seed, and is stored as a first-class receipt arm.
     retest=None
     if paired.get('ratio_median_round',0.0)>1.01:
      retest=measure(cfg,s,a.rounds,a.samples,baseline,candidate,seed_offset=1000003)
     row['timing']={'t_B_norm':measure_single(cfg,s,a.rounds,a.samples,norm_pair,'separate_norm'),
                    't_G_attention_only':measure_single(cfg,s,a.rounds,a.samples,raw_attention,'attention_without_norm'),
                    't_F_unfused':paired_arm_record(paired,'baseline','unfused_norm_plus_attention'),
                    't_F_fused':paired_arm_record(paired,'candidate','fused_norm_prologue_attention'),
                    't_F_pair':paired,
                    'vendor_sdpa':measure_single(cfg,s,a.rounds,a.samples,external,'vendor_sdpa')}
     if retest is not None: row['timing']['t_F_retest']=retest
    except Exception as e:
     row['errors'].append({'stage':'timing','error':repr(e)})
     row['timing']={}
    if ext is not None:
     try: row['timing']['torchinductor_compiled']=measure_single(cfg,s,a.rounds,a.samples,ext,'torchinductor_compiled_sdpa')
     except Exception as e: row['torchinductor_runtime_error']=repr(e)
    row['torchinductor_status']=compiled_status
    row['traces']=[]
    for label,fn in [('unfused',baseline),('fused',candidate),('vendor_sdpa',external),('attention_only',raw_attention)]:
     try:
      path=out.parent/'traces'/name/f'causal_{int(causal)}_S{s}_{label}.chrome_trace.json.gz'; trace(fn,cfg,s,path,False); row['traces'].append(str(path))
     except Exception as e: row.setdefault('errors',[]).append({'stage':f'trace_{label}','error':repr(e)})
    if ext is not None:
     try:
      path=out.parent/'traces'/name/f'causal_{int(causal)}_S{s}_torchinductor.chrome_trace.json.gz'; trace(ext,cfg,s,path,False); row['traces'].append(str(path))
     except Exception as e: row['torchinductor_trace_error']=repr(e)
    row['gpu_exclusivity']['observations']=monitor.finish()
    row['gpu_exclusivity']['after']=gpu_exclusivity_snapshot()
    row['publication_gate']=publication_gate(row,a.rounds,a.samples)
    row['status']='tested_pass' if row['publication_gate']['pass'] else 'tested_invalid'
    rows.append(row); print('DONE',name,causal,s,row['status'],flush=True)
    rec['models'].setdefault(name,{'geometry':{'Hq':cfg[0],'Hkv':cfg[1],'D':cfg[2],'normalization':'per_head_RMSNorm','eps':EPS},'rows':[]})['rows'].append(row)
    rec['status']='running'; out.write_text(json.dumps(rec,indent=2));
 rec_rows=[r for m in rec['models'].values() for r in m['rows']]
 rec['status']='tested_pass' if rec_rows and all(r.get('publication_gate',{}).get('pass') is True and r.get('status')=='tested_pass' for r in rec_rows) else 'tested_invalid'
 rec['publication_eligible']=rec['status']=='tested_pass'
 out.write_text(json.dumps(rec,indent=2)); print('COMPLETE',rec['status'],flush=True)


# H100 refinement APIs and development-only verification.
import sys
from fusion_harness import paired as device_paired, compiler_evidence
@triton.jit
def prepare_factored_kernel(Q,K,GQ,GK,INV,KN,
                            QR:tl.constexpr,KR:tl.constexpr,D:tl.constexpr,R:tl.constexpr):
    tl.static_assert(D==128 or D==256)
    pid=tl.program_id(0);j=tl.arange(0,D)
    if pid<tl.cdiv(QR,R):
        row=pid*R+tl.arange(0,R)
        x=tl.load(Q+row[:,None]*D+j[None,:],row[:,None]<QR,0).to(tl.float32)  # noqa: TR003 -- exact D columns; row tails are masked
        inverse=tl.rsqrt(tl.sum(x*x,1)/D+1e-6)
        tl.store(INV+row,inverse,row<QR)
    else:
        row=(pid-tl.cdiv(QR,R))*R+tl.arange(0,R)
        x=tl.load(K+row[:,None]*D+j[None,:],row[:,None]<KR,0).to(tl.float32)  # noqa: TR003 -- exact D columns; row tails are masked
        gq=tl.load(GQ+j).to(tl.float32)  # noqa: TR003 -- gamma has exactly D elements
        gk=tl.load(GK+j).to(tl.float32)  # noqa: TR003 -- gamma has exactly D elements
        inverse=tl.rsqrt(tl.sum(x*x,1)/D+1e-6)
        value=(x*inverse[:,None]*gk[None,:]*gq[None,:]).to(tl.bfloat16)
        tl.store(KN+row[:,None]*D+j[None,:],value,row[:,None]<KR)


@triton.jit
def factored_attention_kernel(Q,K,V,INV,O,
                              S:tl.constexpr,D:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
                              CAUSAL:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    tl.static_assert(D==128 or D==256)
    pid=tl.program_id(0);h=tl.program_id(1);b=tl.program_id(2);kh=h//(HQ//HK)
    mi=pid*BM+tl.arange(0,BM);j=tl.arange(0,D)
    qrow=b*HQ*S+h*S+mi
    q=tl.load(Q+qrow[:,None]*D+j[None,:],mi[:,None]<S,0)  # noqa: TR003 -- exact D columns; sequence rows are masked
    temperature=tl.load(INV+qrow,mi<S,1)*(D**-.5*1.4426950408889634)
    acc=tl.zeros((BM,D),tl.float32);maximum=tl.full((BM,),float('-inf'),tl.float32)
    denominator=tl.zeros((BM,),tl.float32)
    end=S
    if CAUSAL:end=tl.minimum((pid+1)*BM,S)
    for start in range(0,end,BN):
        ni=start+tl.arange(0,BN);base=(b*HK*S+kh*S+ni)*D
        key=tl.load(K+base[:,None]+j[None,:],ni[:,None]<S,0)  # noqa: TR003 -- exact D columns; sequence rows are masked
        scores=tl.dot(q,tl.trans(key),input_precision='ieee')*temperature[:,None]  # noqa: TR011 -- modern precision API
        scores=tl.where(ni[None,:]<S,scores,float('-inf'))
        if CAUSAL:scores=tl.where(mi[:,None]>=ni[None,:],scores,float('-inf'))
        newmax=tl.maximum(maximum,tl.max(scores,1));alpha=tl.exp2(maximum-newmax)
        probability=tl.exp2(scores-newmax[:,None])
        denominator=denominator*alpha+tl.sum(probability,1);acc=acc*alpha[:,None]
        value=tl.load(V+base[:,None]+j[None,:],ni[:,None]<S,0)  # noqa: TR003 -- exact D columns; sequence rows are masked
        acc=tl.dot(probability.to(tl.bfloat16),value,acc,input_precision='ieee')  # noqa: TR011 -- modern precision API
        maximum=newmax
    tl.store(O+qrow[:,None]*D+j[None,:],(acc/denominator[:,None]).to(tl.bfloat16),mi[:,None]<S)


def launch_factored_attention(inputs,out,inv,kn,*,causal=False,bm=64,bn=64):
    """Two-kernel no-RoPE launch, with preallocated FP32 INV [B,Hq,S].

    Inputs use the legacy five-tensor Q,K,V,GQ,GK order. KN is BF16 like K.
    Gamma vectors are shared across heads, as in the historical model shapes.
    """
    q,k,v,gq,gk=inputs[:5];b,h,s,d=q.shape
    qr=q.numel()//d;kr=k.numel()//d
    prepare_factored_kernel[(triton.cdiv(qr,4)+triton.cdiv(kr,4),)](
        q,k,gq,gk,inv,kn,qr,kr,d,4,num_warps=4)
    return factored_attention_kernel[(triton.cdiv(s,bm),h,b)](
        q,kn,v,inv,out,s,d,h,k.shape[1],causal,bm,bn,num_warps=4,num_stages=1)


def verify_factored_refinement(args):
    import e1_rope_publication_v1 as attention
    args.out.mkdir(parents=True, exist_ok=True)
    census = json.loads(args.census.read_text())
    rec = {'environment': {'triton': triton.__version__, 'torch': torch.__version__}, 'rows': [], 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    mon = attention.ExclusiveMonitor()
    rec['gpu_before'] = attention.gpu_snapshot()
    mon.start()
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for r in census['rows']:
            if r['rotary'] or r['model'] not in ('Gemma-3-4B', 'Gemma-3-12B'):
                continue
            model = r['model']
            b, s, causal = r['shape']
            xs = attention.refinement_inputs(model, b, s, 1909, False)
            ref = attention.refinement_oracle(xs, causal, False)
            arms = {}
            outs = {}
            checks = {}
            evidence = {}
            for tag, label in [('previous_fused', r['selected_fused']), ('unfused', r['selected_unfused'])]:
                mode, bm, bn, gather = label.split('_')
                bm = int(bm)
                bn = int(bn)
                gather = bool(int(gather[1:]))
                out = torch.empty_like(xs[0])
                qn = torch.empty_like(xs[0])
                kn = torch.empty_like(xs[1])
                outs[tag] = out
                arms[tag] = lambda out=out, qn=qn, kn=kn, mode=mode, bm=bm, bn=bn, gather=gather: attention.launch_attention(xs, out, causal=causal, mode=mode, bm=bm, bn=bn, gather=gather, rope=False, hf_norm=False, qn=qn, kn=kn)
            for bm, bn in ((64, 64), (64, 128), (32, 64)):
                tag = f'factored_{bm}_{bn}'
                out = torch.empty_like(xs[0])
                kn = torch.empty_like(xs[1])
                inv = torch.empty(xs[0].shape[:-1], device='cuda', dtype=torch.float32)
                outs[tag] = out
                arms[tag] = lambda out=out, inv=inv, kn=kn, bm=bm, bn=bn: launch_factored_attention(xs, out, inv, kn, causal=causal, bm=bm, bn=bn)
            for tag, call in list(arms.items()):
                compiled = call()
                checks[tag] = attention.metric(outs[tag], ref)
                evidence[tag] = compiler_evidence(compiled, args.out / 'compiler', f"{len(rec['rows'])}_{tag}")
                if not checks[tag]['pass']:
                    del arms[tag]
            assert checks['unfused']['pass'] and checks['previous_fused']['pass']
            candidates = [n for n in arms if n.startswith('factored')]
            if not candidates:
                raise AssertionError((model, s, checks))
            initial = device_paired(arms, rounds=3, samples=5, replays=20, seed=11109)
            selected = min(candidates, key=lambda n: initial['arms'][n]['median_us'])
            retest = device_paired({n: arms[n] for n in ('previous_fused', 'unfused', selected)}, rounds=5, samples=7, replays=20, seed=12109)
            seed_checks = []
            _, bm, bn = selected.split('_')
            for seed in (2909, 3909):
                ys = attention.refinement_inputs(model, b, s, seed, False)
                reference = attention.refinement_oracle(ys, causal, False)
                out = torch.empty_like(ys[0])
                kn = torch.empty_like(ys[1])
                inv = torch.empty(ys[0].shape[:-1], device='cuda', dtype=torch.float32)
                launch_factored_attention(ys, out, inv, kn, causal=causal, bm=int(bm), bn=int(bn))
                check = attention.metric(out, reference)
                assert check['pass'], (model, s, seed, check)
                seed_checks.append({'seed': seed, **check})
            row = {'model': model, 'shape': r['shape'], 'correctness': checks, 'compiler': evidence, 'selected': selected, 'initial': initial, 'retest': retest, 'additional_seed_checks': seed_checks}
            rec['rows'].append(row)
            (args.out / 'results.json').write_text(json.dumps(rec, indent=2) + '\n')
            t = retest['arms']
            print(len(rec['rows']), model, r['shape'], selected, 'vs unfused', round(t['unfused']['median_us'] / t[selected]['median_us'], 3), flush=True)
        rec['tail_checks'] = []
        for s in (33, 129):
            for causal in (False, True):
                xs = attention.refinement_inputs('Gemma-3-4B', 1, s, 13109, False)
                ref = attention.refinement_oracle(xs, causal, False)
                out = torch.empty_like(xs[0])
                kn = torch.empty_like(xs[1])
                inv = torch.empty(xs[0].shape[:-1], device='cuda', dtype=torch.float32)
                launch_factored_attention(xs, out, inv, kn, causal=causal, bn=64)
                check = attention.metric(out, ref)
                assert check['pass'], check
                rec['tail_checks'].append({'s': s, 'causal': causal, **check})
        rec['status'] = 'passed'
    finally:
        rec['gpu_observations'] = mon.finish()
        rec['gpu_after'] = attention.gpu_snapshot()
        (args.out / 'results.json').write_text(json.dumps(rec, indent=2) + '\n')

if __name__ == '__main__':
    if '--refine' in sys.argv:
        parser=argparse.ArgumentParser(description='H100 no-RoPE scalar factoring')
        parser.add_argument('--refine',action='store_true')
        parser.add_argument('--out',type=Path,required=True)
        parser.add_argument('--census',type=Path,required=True)
        verify_factored_refinement(parser.parse_args())
    else:
        main()
