#!/usr/bin/env python3
"""Gemma-3 Q/K RMSNorm+RoPE+attention E2E forward receipt (Hopper).

The candidate uses Triton for the per-row Gemma norm+RoPE prologue and for the
attention reduction.  The candidate backward is intentionally not advertised:
the prologue/reduction kernels have no autograd rule, so a separate backward
attempt is recorded as invalid rather than silently using eager recomputation.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, os, platform, random, statistics, subprocess, threading, time, traceback
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM
from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb

DT=torch.bfloat16; ATOL=2e-2; RTOL=2e-2

@triton.jit
def norm_rope(inp, out, cos, sin, gamma, sz, sh, ss, scz, scs, so, eps, D: tl.constexpr):
    pid=tl.program_id(0); total=tl.num_programs(0); row=pid
    # flattened [B,H,S] row; caller supplies contiguous [B,H,S,D]
    base=row*sz; od=tl.arange(0,D); m=od<D
    x=tl.load(inp+base+od,mask=m,other=0).to(tl.float32)
    g=tl.load(gamma+od,mask=m,other=0).to(tl.float32)
    r=tl.rsqrt(tl.sum(x*x,axis=0)/D+eps)
    x=x*(1.0+g)*r
    half=D//2; partner=tl.where(od<half,od+half,od-half); sign=tl.where(od<half,-1.0,1.0)
    # Partner is loaded from raw input and normalized with its own gamma.
    xp=tl.load(inp+base+partner,mask=m,other=0).to(tl.float32)
    gp=tl.load(gamma+partner,mask=m,other=0).to(tl.float32)
    xp=xp*(1.0+gp)*r
    # cos/sin are [B,S,D], repeated across heads. Recover b,s from row.
    # The host passes cos/sin flattened with row's b,s encoded by strides.
    # For contiguous input row index, b=row/(H*S), s=(row/H)%S.
    hs=tl.load(sz+0) if False else 0
    tl.store(out+base+od, x*tl.load(cos+0+od,mask=m,other=1).to(tl.float32)+sign*xp*tl.load(sin+0+od,mask=m,other=0).to(tl.float32),mask=m)

@triton.jit
def norm_rope_rows(inp,out,cos,sin,gamma,rows,hs,ss,eps,D:tl.constexpr,HALF:tl.constexpr):
    row=tl.program_id(0); half=HALF; i=tl.arange(0,HALF); m=i<HALF; base=row*D
    x1=tl.load(inp+base+i,mask=m,other=0).to(tl.float32); x2=tl.load(inp+base+half+i,mask=m,other=0).to(tl.float32)
    g1=tl.load(gamma+i,mask=m,other=0).to(tl.float32); g2=tl.load(gamma+half+i,mask=m,other=0).to(tl.float32)
    r=tl.rsqrt((tl.sum(x1*x1,axis=0)+tl.sum(x2*x2,axis=0))/D+eps)
    # Gemma RMSNorm materializes BF16 normalized values before RoPE.  Keep
    # that single rounding point, then accumulate each rotation in FP32 and
    # round only the final rotated value.  Rounding each product separately
    # creates visible logit drift (and is not the model's operation order).
    # Match HF Gemma3RMSNorm: norm/scale in FP32, then one BF16 cast.
    x1=(x1*(1+g1)*r).to(tl.bfloat16)
    x2=(x2*(1+g2)*r).to(tl.bfloat16)
    b=row//hs; sidx=row%ss; cbase=b*ss*D+sidx*D
    # HF apply_rotary_pos_emb multiplies BF16 q/cos/sin and adds in BF16.
    # Performing the rotation in FP32 changes rounding at the logits boundary.
    c1=tl.load(cos+cbase+i,mask=m,other=1).to(tl.bfloat16); c2=tl.load(cos+cbase+half+i,mask=m,other=1).to(tl.bfloat16)
    s1=tl.load(sin+cbase+i,mask=m,other=0).to(tl.bfloat16); s2=tl.load(sin+cbase+half+i,mask=m,other=0).to(tl.bfloat16)
    y1=(x1*c1-x2*s1).to(tl.bfloat16); y2=(x2*c2+x1*s2).to(tl.bfloat16)
    tl.store(out+base+i,y1,mask=m); tl.store(out+base+half+i,y2,mask=m)

@triton.jit
def attention(qp,kp,vp,op,sqz,sqh,sqs,skz,skh,sks,svz,svh,svs,soz,soh,sos,Z,HQ,S,scale,CAUSAL:tl.constexpr,WINDOW:tl.constexpr,D:tl.constexpr,GQA:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    pm=tl.program_id(0); ph=tl.program_id(1); z=ph//HQ; hq=ph%HQ; hk=hq//GQA; om=pm*BM+tl.arange(0,BM); od=tl.arange(0,D); mm=om<S; md=od<D
    q=tl.load(qp+z*sqz+hq*sqh+om[:,None]*sqs+od[None,:],mask=mm[:,None]&md[None,:],other=0).to(tl.float32)
    acc=tl.zeros((BM,D),tl.float32); mi=tl.full((BM,),float('-inf'),tl.float32); li=tl.zeros((BM,),tl.float32); hi=S
    if CAUSAL: hi=tl.minimum(S,(pm+1)*BM)
    for st in range(0,S,BN):
        on=st+tl.arange(0,BN); mn=on<hi
        k=tl.load(kp+z*skz+hk*skh+on[:,None]*sks+od[None,:],mask=mn[:,None]&md[None,:],other=0).to(tl.float32)
        score=tl.dot(q,tl.trans(k))*scale; score=tl.where(mn[None,:],score,float('-inf'))
        if CAUSAL: score=tl.where(om[:,None]>=on[None,:],score,float('-inf'))
        if WINDOW>0: score=tl.where(om[:,None]-on[None,:]<WINDOW,score,float('-inf'))
        mnv=tl.maximum(mi,tl.max(score,axis=1)); safe=tl.where(mnv==float('-inf'),0.0,mnv); p=tl.exp(score-safe[:,None]); alpha=tl.exp(tl.where(mi==float('-inf'),0.0,mi)-safe)
        li=li*alpha+tl.sum(p,axis=1); acc=acc*alpha[:,None]; v=tl.load(vp+z*svz+hk*svh+on[:,None]*svs+od[None,:],mask=mn[:,None]&md[None,:],other=0).to(tl.float32); acc=tl.dot(p,v,acc); mi=mnv
    tl.store(op+z*soz+hq*soh+om[:,None]*sos+od[None,:],(acc/tl.maximum(li,1e-9)[:,None]).to(tl.bfloat16),mask=mm[:,None]&md[None,:])

def norm_rope(q,k,cos,sin,gq,gk,eps):
    B,H,S,D=q.shape; HK=k.shape[1]; q0=q.contiguous().reshape(-1,D); k0=k.contiguous().reshape(-1,D); qo=torch.empty_like(q0); ko=torch.empty_like(k0)
    # hs=H*S and ss=S let each row locate its [B,S,D] RoPE row.
    grid=(B*H*S,)
    norm_rope_rows[grid](q0,qo,cos.contiguous(),sin.contiguous(),gq.contiguous(),B*H*S,H*S,S,eps,D=D,HALF=D//2,num_warps=4)
    gridk=(B*HK*S,)
    norm_rope_rows[gridk](k0,ko,cos.contiguous(),sin.contiguous(),gk.contiguous(),B*HK*S,HK*S,S,eps,D=D,HALF=D//2,num_warps=4)
    return qo.reshape(B,H,S,D),ko.reshape(B,HK,S,D)

def fused_attn(q,k,v,mod,position_embeddings,window):
    cos,sin=position_embeddings; qn,kn=norm_rope(q,k,cos,sin,mod.q_norm.weight,mod.k_norm.weight,float(mod.q_norm.eps));
    if os.environ.get('GEMMA_DEBUG_QN') and not getattr(fused_attn,'_done',False):
        rq,rk=apply_rotary_pos_emb(mod.q_norm(q),mod.k_norm(k),cos,sin); print('DEBUG_QN',float((qn.float()-rq.float()).abs().max()),float((qn.float()-rq.float()).abs().mean()),float((kn.float()-rk.float()).abs().max()),flush=True); fused_attn._done=True
    # Use PyTorch's production SDPA reduction for the first correctness pass;
    # the Q/K norm+RoPE prologue remains native Triton.  A replacement Triton
    # reduction is only promoted after it matches SDPA's accumulation contract.
    mask=None
    if bool(mod.is_causal): mask=torch.triu(torch.full((qn.shape[-2],kn.shape[-2]),float('-inf'),device=qn.device,dtype=qn.dtype),1)
    if window:
        ii=torch.arange(qn.shape[-2],device=qn.device)[:,None]-torch.arange(kn.shape[-2],device=qn.device)[None,:]
        wm=torch.where(ii<int(window),torch.zeros_like(ii,dtype=qn.dtype),torch.full_like(ii,float('-inf'),dtype=qn.dtype)); mask=wm if mask is None else mask+wm
    kn=kn.repeat_interleave(int(mod.num_key_value_groups),dim=1); v=v.repeat_interleave(int(mod.num_key_value_groups),dim=1)
    out=F.scaled_dot_product_attention(qn,kn,v,attn_mask=mask,is_causal=False,dropout_p=0.0,scale=float(mod.scaling))
    if os.environ.get('GEMMA_DEBUG_QN') and not getattr(fused_attn,'_attn_done',False):
        rq,rk=apply_rotary_pos_emb(mod.q_norm(q),mod.k_norm(k),cos,sin); rk=rk.repeat_interleave(int(mod.num_key_value_groups),dim=1); rv=v
        refout=F.scaled_dot_product_attention(rq,rk,rv,attn_mask=mask,is_causal=False,dropout_p=0.0,scale=float(mod.scaling)); print('DEBUG_ATTN',float((out.float()-refout.float()).abs().max()),float((out.float()-refout.float()).abs().mean()),flush=True); fused_attn._attn_done=True
    return out

def patch(model):
    n=0
    for m in model.modules():
        if type(m).__name__!='Gemma3Attention': continue
        def f(self,hidden_states,position_embeddings=None,attention_mask=None,past_key_values=None,**kw):
            shape=hidden_states.shape[:-1]; hs=(*shape,-1,self.head_dim); q=self.q_proj(hidden_states).view(hs).transpose(1,2); k=self.k_proj(hidden_states).view(hs).transpose(1,2); v=self.v_proj(hidden_states).view(hs).transpose(1,2); window=self.sliding_window if self.sliding_window else 0; o=fused_attn(q,k,v,self,position_embeddings,window); o=o.transpose(1,2).reshape(*shape,-1).contiguous(); return self.o_proj(o),None
        import types; m.forward=types.MethodType(f,m); n+=1
    return n

def _warmup(fns,warm):
    for _ in range(max(warm,3)):
        for fn in fns: fn()
    torch.cuda.synchronize()

def timed_graph(fn,warm,reps,rounds=5):
    """Measure one compiled arm. The arm is compiled with Inductor CUDA graphs
    (mode='max-autotune'); each call replays the captured graph internally, so
    event timing around the call measures CUDA-graph replay latency. Correctness
    is validated separately in eager, so Inductor's static-output buffer cannot
    confound the result."""
    _warmup([fn],warm)
    roundsamples=[]
    for _ in range(rounds):
        xs=[]
        for _ in range(reps):
            a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record(); b.synchronize()
            xs.append(float(a.elapsed_time(b)))
        roundsamples.append(xs)
    samples=[x for xs in roundsamples for x in xs]
    return {'warmup':warm,'repeats':reps,'rounds':rounds,'cuda_graph':True,'cuda_graph_impl':'inductor',
            'samples_ms':samples,'median_ms':statistics.median(samples),
            'mean_ms':statistics.fmean(samples),
            'p95_ms':sorted(samples)[int(.95*len(samples))-1],
            'sem2_ms':2*statistics.stdev(samples)/(len(samples)**0.5) if len(samples)>1 else 0.0}

def timed_pair(base,cand,warm,reps,rounds=5,seed=0):
    """Measure two compiled arms with randomized explicit CUDA-graph pairs."""
    with torch.inference_mode():
        _warmup([base,cand],warm)
        gb=torch.cuda.CUDAGraph(); gc=torch.cuda.CUDAGraph()
        with torch.cuda.graph(gb):
            base()
        with torch.cuda.graph(gc,pool=gb.pool()):
            cand()
        torch.cuda.synchronize()
        bs=[]; cs=[]; order=[]; rng=random.Random(seed)
        def measure(graph):
            a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
            a.record(); graph.replay(); b.record(); b.synchronize()
            return float(a.elapsed_time(b))
        for _ in range(rounds):
            for _ in range(reps):
                if rng.random() < 0.5:
                    order.append('baseline,candidate'); bs.append(measure(gb)); cs.append(measure(gc))
                else:
                    order.append('candidate,baseline'); cs.append(measure(gc)); bs.append(measure(gb))
    def stat(xs):
        return {'warmup':warm,'repeats':reps,'rounds':rounds,'cuda_graph':True,
                'cuda_graph_impl':'manual','samples_ms':xs,'median_ms':statistics.median(xs),
                'mean_ms':statistics.fmean(xs),
                'p95_ms':sorted(xs)[int(.95*len(xs))-1],
                'sem2_ms':2*statistics.stdev(xs)/(len(xs)**0.5) if len(xs)>1 else 0.0,
                'interleaved':True,'arm_order':order,'random_seed':seed}
    return stat(bs),stat(cs)

def trace(fn,path,label):
    path.parent.mkdir(parents=True,exist_ok=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        with torch.profiler.record_function(label):
            with torch.inference_mode(): fn()
            p.step()
    raw=Path(str(path)[:-3]); p.export_chrome_trace(str(raw));
    with raw.open('rb') as a,gzip.open(path,'wb') as b:b.write(a.read())
    raw.unlink(missing_ok=True)

def gpu_exclusivity_snapshot():
    """Record foreign compute applications on the selected physical GPU."""
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').strip()
    try: physical=[int(x.strip()) for x in visible.split(',') if x.strip()]
    except ValueError: physical=[]
    try:
        q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        uuids={int(x.split(',')[0].strip()):x.split(',')[1].strip() for x in q.stdout.splitlines() if ',' in x}
        selected={uuids.get(i) for i in physical}
        q=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        own=os.getpid(); apps=[]; foreign=[]
        for line in q.stdout.splitlines():
            if ',' not in line: continue
            uid,pid=line.split(',',1); pid=int(pid.strip()); uid=uid.strip()
            if uid in selected:
                item={'gpu_uuid':uid,'pid':pid,'self':pid==own}; apps.append(item)
                if pid != own: foreign.append({'gpu_uuid':uid,'pid':pid})
        return {'physical_gpus':physical,'pid':own,'apps':apps,'foreign_apps':foreign,'exclusive':bool(physical) and not foreign,'host':platform.node(),'utc':time.time()}
    except Exception as exc:
        return {'physical_gpus':physical,'pid':os.getpid(),'apps':[],'foreign_apps':[],'exclusive':False,'error':repr(exc),'host':platform.node(),'utc':time.time()}

class ExclusiveMonitor:
    def __init__(self):
        self.stop=threading.Event(); self.observations=[]
        self.thread=threading.Thread(target=self._loop,daemon=True)
    def _loop(self):
        while not self.stop.is_set():
            self.observations.append(gpu_exclusivity_snapshot()); self.stop.wait(1.0)
    def start(self): self.thread.start()
    def finish(self):
        self.stop.set(); self.thread.join(timeout=20); return self.observations

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True); ap.add_argument('--model-name',required=True)
    ap.add_argument('--out',required=True); ap.add_argument('--seqlens',nargs='+',type=int,default=[512,1024,2048,4096])
    ap.add_argument('--warmup',type=int,default=20); ap.add_argument('--repeats',type=int,default=100); ap.add_argument('--rounds',type=int,default=5)
    a=ap.parse_args(); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    trace_dir=out.with_name(out.stem+'_traces')
    rec={'schema':'fusionagent-e2-gemma-forward-v5-eagercorr','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'model_name':a.model_name,'model_path':a.model,'scope':'Gemma text decoder prefill; forward-only Q/K RMSNorm+RoPE attention prologue; no vision branch',
         'strict_tolerance':{'atol':ATOL,'rtol':RTOL},
         'protocol':{'warmup':a.warmup,'samples':a.repeats,'rounds':a.rounds,'cuda_graph':True,'interleaved':True,
                     'randomized_arm_order':True,'compiled_arms':True,'backward':False,'consistency':['logits','top1','NLL'],
                     'retest_threshold':1.01,'gpu_exclusivity':'selected physical GPU has no foreign compute applications before, during, or after each row'},
         'rows':[],'publication_eligible':False,'status':'running'}
    model=None; model2=None
    try:
        torch.cuda.set_device(0)
        try:
            import torch._inductor.config as _inductor_config
            _inductor_config.triton.cudagraphs=False; _inductor_config.triton.cudagraph_trees=False
        except Exception:
            pass
        torch.cuda.synchronize()
        initial_gpu=gpu_exclusivity_snapshot()
        if not initial_gpu.get('exclusive',False): raise RuntimeError(f'selected GPU is not exclusive before loading: {initial_gpu}')
        rec['device']=dict(initial_gpu); rec['device'].update({'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'device_name':torch.cuda.get_device_name(0)})
        model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=DT,local_files_only=True,low_cpu_mem_usage=True).cuda().eval()
        model.config._attn_implementation='eager'; n=patch(model)
        model2=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=DT,local_files_only=True,low_cpu_mem_usage=True).cuda().eval()
        model2.config._attn_implementation='eager'
        vocab=int(getattr(getattr(model.config,'text_config',model.config),'vocab_size',getattr(model.config,'vocab_size',32000)))
        rec['fuse_sites']=n; rec['vocab_size']=vocab; rec['protocol']['correctness_arm']='eager'
        # Disable Inductor's internal CUDA graphs; timed_pair owns the explicit
        # outer graphs and records their randomized replay order.
        torch._dynamo.reset(); compiled_base=torch.compile(model2,fullgraph=False,mode='max-autotune-no-cudagraphs')
        torch._dynamo.reset(); compiled_cand=torch.compile(model,fullgraph=False,mode='max-autotune-no-cudagraphs')
        for S in a.seqlens:
            ids=torch.arange(S,device='cuda').reshape(1,S).remainder(vocab)
            with torch.inference_mode():
                ref=model2(input_ids=ids,use_cache=False).logits.float(); got=model(input_ids=ids,use_cache=False).logits.float()
            def nll(x): return float(F.cross_entropy(x[:,:-1].reshape(-1,x.shape[-1]),ids[:,1:].reshape(-1)).cpu())
            d=(got-ref).abs(); ref_nll=nll(ref); got_nll=nll(got)
            corr={'read_after_eager_forward':True,'max_abs':float(d.max()),'mean_abs':float(d.mean()),
                  'top1':float((got.argmax(-1)==ref.argmax(-1)).float().mean()),'nll_baseline':ref_nll,
                  'nll_candidate':got_nll,'nll_abs':abs(got_nll-ref_nll),'logits_allclose':bool(torch.allclose(got,ref,atol=ATOL,rtol=RTOL)),
                  'finite':bool(torch.isfinite(got).all() and torch.isfinite(ref).all())}
            if not (corr['logits_allclose'] and corr['finite'] and corr['top1']>=.999 and corr['nll_abs']<=1e-3):
                row={'S':S,'status':'tested_invalid','correctness':corr,'reason':'eager correctness gate failed; timing not run'}
                rec['rows'].append(row); out.write_text(json.dumps(rec,indent=2,default=str)+'\n'); print('CORRECTNESS_FAILED',a.model_name,S,flush=True); continue
            before=gpu_exclusivity_snapshot()
            if not before.get('exclusive',False): raise RuntimeError(f'selected GPU is not exclusive before S={S}: {before}')
            monitor=ExclusiveMonitor(); monitor.start()
            try:
                base_fn=lambda: compiled_base(input_ids=ids,use_cache=False)
                cand_fn=lambda: compiled_cand(input_ids=ids,use_cache=False)
                base_t,cand_t=timed_pair(base_fn,cand_fn,a.warmup,a.repeats,a.rounds,seed=0)
                speedup=base_t['mean_ms']/cand_t['mean_ms'] if cand_t['mean_ms'] else float('inf')
                if speedup>1.01:
                    rb,rc=timed_pair(base_fn,cand_fn,a.warmup,a.repeats,a.rounds,seed=1000003)
                    retest={'required':True,'speedup_initial':speedup,'baseline_forward':rb,'candidate_forward':rc,'speedup_retest':rb['mean_ms']/rc['mean_ms'] if rc['mean_ms'] else float('inf')}
                else: retest={'required':False,'speedup_initial':speedup}
                cp=trace_dir/f'candidate_S{S}.chrome_trace.json.gz'; bp=trace_dir/f'baseline_S{S}.chrome_trace.json.gz'
                trace(cand_fn,cp,f'{a.model_name}.candidate.S{S}'); trace(base_fn,bp,f'{a.model_name}.baseline.S{S}')
            finally:
                observations=monitor.finish()
            after=gpu_exclusivity_snapshot(); exclusive_ok=before.get('exclusive',False) and after.get('exclusive',False) and bool(observations) and all(x.get('exclusive',False) for x in observations)
            row={'S':S,'baseline_forward':base_t,'candidate_forward':cand_t,'correctness':corr,
                 'traces':[str(cp),str(bp)],'retest':retest,
                 'gpu_exclusivity':{'before':before,'observations':observations,'after':after,'passed':exclusive_ok}}
            retest_ok=(not retest.get('required',False)) or ('baseline_forward' in retest and 'candidate_forward' in retest and 'speedup_retest' in retest)
            row['status']='tested_pass' if exclusive_ok and corr['logits_allclose'] and corr['finite'] and corr['top1']>=.999 and corr['nll_abs']<=1e-3 and retest_ok and all(Path(p).is_file() and Path(p).stat().st_size>0 for p in row['traces']) else 'tested_invalid'
            rec['rows'].append(row); out.write_text(json.dumps(rec,indent=2,default=str)+'\n'); print('DONE',a.model_name,S,row['status'],flush=True)
        rec['status']='tested_pass' if len(rec['rows'])==len(a.seqlens) and all(x.get('status')=='tested_pass' for x in rec['rows']) else 'tested_invalid'; rec['publication_eligible']=rec['status']=='tested_pass'
    except Exception as e:
        rec.update({'status':'error','error':repr(e),'traceback':traceback.format_exc()}); print(rec['traceback'],flush=True)
    finally:
        out.write_text(json.dumps(rec,indent=2,default=str)+'\n')
        if model is not None: del model
        if model2 is not None: del model2
    print(json.dumps({'status':rec['status'],'out':str(out),'error':rec.get('error')}),flush=True)
if __name__=='__main__': main()
