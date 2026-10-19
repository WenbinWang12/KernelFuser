#!/usr/bin/env python3
"""Strict E9 Hopper-equivalent OLMoE packed-MoE experiment (H100/H200 single GPU).

The only changed operation is the pointwise ``SiLU(gate) * up`` in the
Transformers eager packed expert module.  Routing, top-k weights, packed
projections, down projections and ``index_add_`` are the unmodified model
implementation.  The receipt records independent K/M/E forward and
backward distributions, per-parameter gradient/update comparisons, and six
Chrome traces.
"""
from __future__ import annotations
import argparse, gc, gzip, hashlib, importlib.util, inspect, json, os, platform, shutil, statistics, subprocess, sys, time, traceback
from pathlib import Path
from typing import Any, Callable
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

ATOL = 2e-2; RTOL = 2e-2; ALLOWED = set(range(0,4)); HARDWARE = "h100"

def write_json(p: Path, x: Any):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(x, indent=2, sort_keys=True, default=str)+"\n")
def sha(p: Path):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
def load_native(p: Path):
    s=importlib.util.spec_from_file_location('fusionagent_native_swiglu_olmoe_v6',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def sync(): torch.cuda.synchronize()
def finite(x): return x is None or bool(torch.isfinite(x.detach()).all().item())
def metric(a,b):
    aa,bb=a.detach().float(),b.detach().float(); d=(aa-bb).abs(); rel=d/bb.abs().clamp_min(1e-12)
    return {'allclose':finite(aa) and finite(bb) and bool(torch.allclose(aa,bb,atol=ATOL,rtol=RTOL)), 'exact_equal':bool(torch.equal(a,b)), 'max_abs':float(d.max().item()) if d.numel() else 0.0,
            'max_rel':float(rel.max().item()) if d.numel() else 0.0, 'numel':int(aa.numel()), 'atol':ATOL, 'rtol':RTOL}
def physical_id():
    v=os.environ.get('CUDA_VISIBLE_DEVICES',''); t=v.split(',')[0].strip()
    allowed = set(range(0,4)) if HARDWARE == 'h100' else set(range(4,8))
    if ',' in v or not t.isdigit() or int(t) not in allowed: raise RuntimeError(f'CUDA_VISIBLE_DEVICES must select one physical {HARDWARE.upper()} GPU ({sorted(allowed)})')
    p=int(t); a=os.environ.get('FUSIONAGENT_PHYSICAL_GPU_ID')
    if a is not None and a.strip()!=str(p): raise RuntimeError('physical attestation mismatch')
    return p
def timed(fn:Callable[[],Any], warmup:int, repeats:int):
    if warmup<20 or repeats<100: raise ValueError('publication timing requires warmup>=20 and repeats>=100')
    for _ in range(warmup): fn()
    sync(); xs=[]
    for _ in range(repeats):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record(); fn(); b.record(); b.synchronize(); xs.append(float(a.elapsed_time(b)))
    return {'warmup':warmup,'repeats':repeats,'samples_ms':xs,'mean_ms':statistics.fmean(xs),'p50_ms':statistics.median(xs),'p95_ms':sorted(xs)[max(0,int(.95*len(xs))-1)],'min_ms':min(xs),'max_ms':max(xs)}
def timed_backward(builder,warmup,repeats):
    if warmup<20 or repeats<100: raise ValueError('publication timing requires warmup>=20 and repeats>=100')
    for _ in range(warmup):
        y,leaves,dy=builder(); torch.autograd.grad(y,leaves,dy,retain_graph=False)
    sync(); xs=[]
    for _ in range(repeats):
        y,leaves,dy=builder(); sync(); a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record(); torch.autograd.grad(y,leaves,dy,retain_graph=False); b.record(); b.synchronize(); xs.append(float(a.elapsed_time(b)))
    return {'warmup':warmup,'repeats':repeats,'samples_ms':xs,'mean_ms':statistics.fmean(xs),'p50_ms':statistics.median(xs),'p95_ms':sorted(xs)[max(0,int(.95*len(xs))-1)],'min_ms':min(xs),'max_ms':max(xs)}
def timed_model_backward(builder,warmup,repeats):
    if warmup<20 or repeats<100: raise ValueError('publication timing requires warmup>=20 and repeats>=100')
    for _ in range(warmup): builder().backward()
    sync(); xs=[]
    for _ in range(repeats):
        loss=builder(); sync(); a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record(); loss.backward(); b.record(); b.synchronize(); xs.append(float(a.elapsed_time(b)))
    return {'warmup':warmup,'repeats':repeats,'samples_ms':xs,'mean_ms':statistics.fmean(xs),'p50_ms':statistics.median(xs),'p95_ms':sorted(xs)[max(0,int(.95*len(xs))-1)],'min_ms':min(xs),'max_ms':max(xs)}
def chrome(fn,p,label):
    p.parent.mkdir(parents=True,exist_ok=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.profiler.record_function(label): fn()
        prof.step()
    raw=p.with_suffix('.json'); prof.export_chrome_trace(str(raw));
    with raw.open('rb') as a,gzip.open(p,'wb') as b: b.write(a.read())
    raw.unlink(missing_ok=True)
def packed_experts(model):
    return [(n,m) for n,m in model.named_modules() if hasattr(m,'gate_up_proj') and hasattr(m,'down_proj') and getattr(m.gate_up_proj,'ndim',0)==3]
def patch_experts(experts,native,counter):
    originals=[]
    for name,ex in experts:
        originals.append((ex,ex.forward))
        def f(hidden_states,top_k_index,top_k_weights,*args,_ex=ex,**kwargs):
            counter['containers_called']+=1; out=torch.zeros_like(hidden_states)
            with torch.no_grad():
                mask=F.one_hot(top_k_index,num_classes=_ex.num_experts).permute(2,1,0); hits=(mask.sum(dim=(-1,-2))>0).nonzero().flatten()
            for ei in hits:
                e=int(ei); pos,tok=torch.where(mask[e]); cur=hidden_states[tok]; z=F.linear(cur,_ex.gate_up_proj[e]); g,u=z.chunk(2,-1); y=native.swiglu(g,u); y=F.linear(y,_ex.down_proj[e]); y=y*top_k_weights[tok,pos,None]; out.index_add_(0,tok,y.to(out.dtype)); counter['fused_launches']+=1; counter['routed_tokens']+=int(tok.numel())
            return out
        ex.forward=f
    return originals
def restore(orig):
    for m,f in orig:m.forward=f
def run_model(model,ids,labels,backward=True):
    model.zero_grad(set_to_none=True); out=model(input_ids=ids,labels=labels if backward else None,use_cache=False,return_dict=True); 
    if backward: out.loss.backward()
    sync(); return {'loss':out.loss.detach().float().cpu().clone() if backward else None,'logits':out.logits.detach().float().cpu().clone(),
        # Keep gradients in their native BF16 representation while staging to
        # host memory.  Every comparison casts one parameter at a time to
        # float32; this avoids the 2x CPU memory blow-up of a full-model
        # float32 clone while retaining elementwise strict checks.
        'grads':{n:(p.grad.detach().cpu() if p.grad is not None else None) for n,p in model.named_parameters()} if backward else {}}
def streamed_metric(a,b):
    if a.shape != b.shape: return {'allclose':False,'shape_mismatch':True}
    aa=a.reshape(-1); bb=b.reshape(-1); out={'allclose':True,'exact_equal':True,'finite':True,'max_abs':0.0,'max_rel':0.0,'numel':a.numel(),'atol':ATOL,'rtol':RTOL}
    for lo in range(0,a.numel(),1<<20):
        x=aa[lo:lo+(1<<20)].to('cuda'); y=bb[lo:lo+(1<<20)].to('cuda'); m=metric(x,y)
        out['allclose'] &= m['allclose']; out['exact_equal'] &= m['exact_equal']; out['finite'] &= finite(x) and finite(y)
        out['max_abs']=max(out['max_abs'],m['max_abs']); out['max_rel']=max(out['max_rel'],m['max_rel'])
    return out
def compare_runs(ref,cand):
    rows={'loss':metric(cand['loss'],ref['loss']),'logits':metric(cand['logits'],ref['logits'])}; gs={}; missing=[]
    for n,a in ref['grads'].items():
        b=cand['grads'].get(n)
        if a is None or b is None:
            if a is not None or b is not None: missing.append(n)
        else: gs[n]=streamed_metric(b,a)
    rows['gradients']={'parameter_count':len(gs),'missing':missing,'finite':all(v['finite'] for v in gs.values()),'allclose':not missing and all(v['allclose'] for v in gs.values()),'max_abs':max((v['max_abs'] for v in gs.values()),default=0.0),'max_rel':max((v['max_rel'] for v in gs.values()),default=0.0),'rows':gs}
    rows['pass']=all(rows[k]['allclose'] for k in ('loss','logits')) and rows['gradients']['allclose'] and rows['gradients']['finite']; return rows
def update_compare(model,ref,cand,lr):
    """Actual in-place SGD on every model Parameter, compare true BF16 deltas.

    Parameters are updated one at a time to bound temporary GPU storage.
    No-momentum/no-weight-decay SGD is separable over parameters, so this is
    the same step as a full parameter group. Both steps start from identical
    weights and restore the checkpoint before timing. Zero/unused gradients
    are recorded explicitly. Delta equality is exact, stricter than .02.
    """
    rows={}; ok=True; changed=0
    model.zero_grad(set_to_none=True)
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        a=ref['grads'].get(n)
        b=cand['grads'].get(n)
        if a is None or b is None: rows[n]={'allclose':a is None and b is None}; ok &= rows[n]['allclose']; continue
        before=p.detach().clone(); opt=torch.optim.SGD([p],lr=lr,foreach=False)
        p.grad=a.to(device=p.device,dtype=p.dtype); opt.step(); after_ref=p.detach().clone()
        with torch.no_grad(): p.copy_(before)
        p.grad=b.to(device=p.device,dtype=p.dtype); opt.step(); after_cand=p.detach()
        # True rounded parameter deltas, not the algebraic -lr*g expression.
        da=after_ref.float()-before.float(); db=after_cand.float()-before.float(); r=metric(db,da)
        r['changed_ref']=bool(torch.any(da!=0)); r['changed_candidate']=bool(torch.any(db!=0)); r['finite']=finite(da) and finite(db)
        r['exact_delta_match']=bool(torch.equal(da,db)); rows[n]=r; ok &= r['finite'] and r['exact_delta_match']; changed += int(r['changed_ref'])
        with torch.no_grad(): p.copy_(before)
        p.grad=None; del before,after_ref,da,db,opt
    return {'status':'tested_pass' if ok and changed else 'tested_invalid','method':'actual torch.optim.SGD sequential parameter steps; exact rounded update delta equality','lr':lr,'parameter_count':len(rows),'changed_parameter_count':changed,'rows':rows}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model-path',type=Path,required=True); ap.add_argument('--model-id',required=True); ap.add_argument('--native',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--seq',type=int,default=8); ap.add_argument('--warmup',type=int,default=20); ap.add_argument('--repeats',type=int,default=100); ap.add_argument('--lr',type=float,default=1e-5); ap.add_argument('--hardware',choices=['h100','h200'],default='h100'); ap.add_argument('--physical-gpu',type=int,required=True); a=ap.parse_args(); globals()['HARDWARE']=a.hardware; os.environ['FUSIONAGENT_PHYSICAL_GPU_ID']=str(a.physical_gpu)
    rec={'schema':'fusionagent-e9-hopper-equivalent-v1','suite_id':'fusionagent-e9-hopper-equivalent-v1','status':'started','model_id':a.model_id,'model_path':str(a.model_path),'backend_variant':'triton_sm90_hopper','native_tlx_status':'not_used_due_triton_3_6_api','publication_eligible':False,'strict_tolerance':{'atol':ATOL,'rtol':RTOL},'requested':vars(a)|{'native':str(a.native)},'chrome_viewer':'chrome://tracing','claim_boundary':'Hopper-equivalent Triton SwiGLU only; original eager MoE routing and projections; no Blackwell tmem claim'}; model=None
    try:
        p=physical_id(); assert p == a.physical_gpu; assert torch.cuda.is_available(); torch.cuda.set_device(0); native=load_native(a.native); rec['provenance']={'runner_sha256':sha(Path(__file__)),'native_source':str(a.native.resolve()),'native_sha256':sha(a.native),'hardware_family':a.hardware,'physical_gpu':p,'physical_gpu_ids':[p],'gpu_policy':f'{a.hardware.upper()} physical GPU; single GPU only; physical attestation required','cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'native_has_triton':bool(hasattr(native,'_Fn') and hasattr(native,'_bwd'))}; torch.manual_seed(20261006); torch.cuda.manual_seed_all(20261006)
        cfg=AutoConfig.from_pretrained(str(a.model_path),local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(str(a.model_path),torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True,trust_remote_code=False,experts_implementation='eager').cuda(); model.train(); experts=packed_experts(model)
        if not experts:
            raise RuntimeError('no packed MoE expert modules')
        rec['module_count']=len(experts); rec['module_names']=[n for n,_ in experts]; print('stage:model_loaded',len(experts),flush=True)
        # Correctness checks use every routed expert container and two expert ids where possible.
        krows=[]; mrows=[]; print('stage:local_start',flush=True)
        # Local module checks request input gradients only.  Disable expert
        # weight leaf gradients for this diagnostic pass; otherwise autograd
        # also materializes every 3-D expert weight gradient on each of the
        # 48 containers, making a small routed check needlessly multi-minute.
        for _, ex0 in experts:
            ex0.gate_up_proj.requires_grad_(False); ex0.down_proj.requires_grad_(False)
        # Packed expert containers share the same SwiGLU kernel contract; the
        # full model pass below still patches and traverses every container.
        # Use one representative here to avoid repeating 48 large expert
        # projections before the publication-scale E2E run.
        for j,(name,ex) in enumerate(experts):
            d=ex.gate_up_proj.device; dt=ex.gate_up_proj.dtype; hidden=int(ex.gate_up_proj.shape[-1]); inter=int(ex.gate_up_proj.shape[-2]//2); nseq=max(2,a.seq); torch.manual_seed(8000+j); x=torch.randn((nseq,hidden),device=d,dtype=dt)
            for ei in sorted(set([0,int(torch.randint(0,int(ex.num_experts),(1,)).item())])):
                wg=ex.gate_up_proj[ei]; wd=ex.down_proj[ei]; gg=F.linear(x,wg).chunk(2,-1); g,u=[q.detach().requires_grad_(True) for q in gg]; dy=torch.randn_like(g)
                y0=F.silu(g); z0=y0*u; yn=native.swiglu(g,u); kg=metric(yn,z0)
                gd0,ud0=torch.autograd.grad(z0,(g,u),dy,retain_graph=True); gdn,udn=native.backward(g,u,dy); kr={'module':name,'expert':ei,'forward':kg,'gate_backward':metric(gdn,gd0),'up_backward':metric(udn,ud0),'finite':finite(gdn) and finite(udn)}; kr['pass']=kr['finite'] and all(kr[x]['allclose'] for x in ('forward','gate_backward','up_backward')); krows.append(kr)
                # Module scope includes the selected expert's input, gate/up
                # and down projection parameters.  Clone only that routed
                # expert rather than materializing a full 3-D MoE gradient.
                xx=x.detach().requires_grad_(True); gw=wg.detach().clone().requires_grad_(True); dw=wd.detach().clone().requires_grad_(True)
                def module_step(candidate):
                    gg,uu=F.linear(xx,gw).chunk(2,-1); yy=native.swiglu(gg,uu) if candidate else F.silu(gg)*uu; return F.linear(yy,dw)
                ref_value=module_step(False); ref_loss=ref_value.float().square().mean(); ref_grads=torch.autograd.grad(ref_loss,(xx,gw,dw),retain_graph=False); ref_value=ref_value.detach().clone()
                xx=x.detach().requires_grad_(True); gw=wg.detach().clone().requires_grad_(True); dw=wd.detach().clone().requires_grad_(True); cand_value=module_step(True); cand_loss=cand_value.float().square().mean(); cand_grads=torch.autograd.grad(cand_loss,(xx,gw,dw),retain_graph=False); cand_value=cand_value.detach().clone()
                mr={'module':name,'expert':ei,'output':metric(cand_value,ref_value),'input_gradient':metric(cand_grads[0],ref_grads[0]),'gate_up_gradient':metric(cand_grads[1],ref_grads[1]),'down_gradient':metric(cand_grads[2],ref_grads[2]),'parameter_scope':'selected routed expert gate_up/down','finite':all(finite(q) for q in cand_grads)}; mr['pass']=mr['finite'] and all(mr[q]['allclose'] for q in ('output','input_gradient','gate_up_gradient','down_gradient')); mrows.append(mr)
        for _, ex0 in experts:
            ex0.gate_up_proj.requires_grad_(True); ex0.down_proj.requires_grad_(True)
        rec['levels']={'kernel':{'status':'tested_pass' if all(x['pass'] for x in krows) else 'tested_invalid','rows':krows},'module':{'status':'tested_pass' if all(x['pass'] for x in mrows) else 'tested_invalid','rows':mrows}}; print('stage:local_done',flush=True)
        vocab=int(getattr(model.config,'vocab_size',32000)); ids=torch.arange(a.seq,device='cuda').reshape(1,a.seq).remainder(vocab); labels=ids.clone(); torch.manual_seed(20261006); torch.cuda.manual_seed_all(20261006); print('stage:e2e_baseline_start',flush=True); ref=run_model(model,ids,labels,True); print('stage:e2e_baseline_done',flush=True); counter={'containers_called':0,'fused_launches':0,'routed_tokens':0}; orig=patch_experts(experts,native,counter)
        try: print('stage:e2e_candidate_start',flush=True); cand=run_model(model,ids,labels,True); print('stage:e2e_candidate_done',flush=True)
        finally: restore(orig)
        cmp=compare_runs(ref,cand); up=update_compare(model,ref,cand,a.lr); rec['levels']['end_to_end']={'status':'tested_pass' if cmp['pass'] and up['status']=='tested_pass' and counter['containers_called']>=len(experts) and counter['routed_tokens']>0 else 'tested_invalid','comparison':cmp,'update_comparison':up,'candidate_evidence':counter,'training_scope':'all trainable parameters','backward_mode':'Hopper Triton SwiGLU'}; print('stage:e2e_correctness_done',cmp['pass'],up['status'],flush=True)
        # representative module and kernel timing, using actual module call for module scope
        name,ex=experts[0]; d=ex.gate_up_proj.device; h=int(ex.gate_up_proj.shape[-1]); inter=int(ex.gate_up_proj.shape[-2]//2); x=torch.randn((a.seq,h),device=d,dtype=ex.gate_up_proj.dtype); g,u=F.linear(x,ex.gate_up_proj[0]).chunk(2,-1); dy=torch.randn_like(g)
        def kb0(): return F.silu(g)*u
        def kbn(): return native.swiglu(g,u)
        def kbb(): gg=g.detach().requires_grad_(True); uu=u.detach().requires_grad_(True); return F.silu(gg)*uu,(gg,uu),dy
        def kbc(): gg=g.detach().requires_grad_(True); uu=u.detach().requires_grad_(True); return native.swiglu(gg,uu),(gg,uu),dy
        topk=torch.zeros((a.seq,1),device=d,dtype=torch.long); weights=torch.ones((a.seq,1),device=d,dtype=ex.gate_up_proj.dtype)
        def mb0(): return ex(x,topk,weights)
        counter_t={'containers_called':0,'fused_launches':0,'routed_tokens':0}; orig=patch_experts([(name,ex)],native,counter_t)
        def mbn(): return ex(x,topk,weights)
        restore(orig)
        def mbb():
            xx=x.detach().requires_grad_(True); y=ex(xx,topk,weights); return y,(xx,),torch.ones_like(y)
        orig=patch_experts([(name,ex)],native,counter_t)
        def mbc():
            xx=x.detach().requires_grad_(True); y=ex(xx,topk,weights); return y,(xx,),torch.ones_like(y)
        restore(orig)
        print('stage:timing_kernel_module_start',flush=True); rec['timing']={'kernel':{'baseline':{'forward':timed(kb0,a.warmup,a.repeats),'backward':timed_backward(kbb,a.warmup,a.repeats)},'candidate':{'forward':timed(kbn,a.warmup,a.repeats),'backward':timed_backward(kbc,a.warmup,a.repeats)}},'module':{'baseline':{'forward':timed(mb0,a.warmup,a.repeats),'backward':timed_backward(mbb,a.warmup,a.repeats)},'candidate':{}}}
        orig=patch_experts([(name,ex)],native,counter_t)
        try:
            rec['timing']['module']['candidate']={'forward':timed(mbn,a.warmup,a.repeats),'backward':timed_backward(mbc,a.warmup,a.repeats)}
        finally: restore(orig)
        print('stage:timing_e2e_start',flush=True)
        def eforward(): model.zero_grad(set_to_none=True); model(input_ids=ids,labels=labels,use_cache=False,return_dict=True); sync()
        def eloss(): model.zero_grad(set_to_none=True); return model(input_ids=ids,labels=labels,use_cache=False,return_dict=True).loss
        def einference():
            with torch.no_grad(): model(input_ids=ids,use_cache=False,return_dict=True)
            sync()
        model.train(); rec['timing']['end_to_end']={'baseline':{'forward_train':timed(eforward,a.warmup,a.repeats),'backward_only':timed_model_backward(eloss,a.warmup,a.repeats)}}; model.eval(); rec['timing']['end_to_end']['baseline']['forward_inference']=timed(einference,a.warmup,a.repeats); model.train(); orig=patch_experts(experts,native,{'containers_called':0,'fused_launches':0,'routed_tokens':0})
        try:
            rec['timing']['end_to_end']['candidate']={'forward_train':timed(eforward,a.warmup,a.repeats),'backward_only':timed_model_backward(eloss,a.warmup,a.repeats)}; model.eval(); rec['timing']['end_to_end']['candidate']['forward_inference']=timed(einference,a.warmup,a.repeats); model.train()
        finally: restore(orig)
        td=a.out.parent/(a.out.stem+'_traces'); model.train(); chrome(lambda:run_model(model,ids,labels,True),td/'baseline_end_to_end.chrome_trace.json.gz','baseline.end_to_end'); orig=patch_experts(experts,native,{'containers_called':0,'fused_launches':0,'routed_tokens':0})
        try: chrome(lambda:run_model(model,ids,labels,True),td/'candidate_end_to_end.chrome_trace.json.gz','candidate.end_to_end')
        finally: restore(orig)
        chrome(kb0,td/'baseline_kernel.chrome_trace.json.gz','baseline.kernel'); chrome(kbn,td/'candidate_kernel.chrome_trace.json.gz','candidate.kernel'); chrome(mb0,td/'baseline_module.chrome_trace.json.gz','baseline.module'); orig=patch_experts([(name,ex)],native,{'containers_called':0,'fused_launches':0,'routed_tokens':0})
        try: chrome(mbn,td/'candidate_module.chrome_trace.json.gz','candidate.module')
        finally: restore(orig)
        rec['traces']={'status':'complete','directory':str(td),'files':sorted(p.name for p in td.glob('*.json.gz')),'count':len(list(td.glob('*.json.gz')))}
        rec['status']='tested_pass' if all(rec['levels'][x]['status']=='tested_pass' for x in ('kernel','module','end_to_end')) and rec['traces']['count']==6 else 'tested_invalid'
    except Exception as e: rec.update({'status':'error','error_type':type(e).__name__,'error':repr(e),'traceback':traceback.format_exc()})
    finally:
        write_json(a.out,rec); del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(json.dumps({'status':rec['status'],'model_id':a.model_id,'out':str(a.out),'error':rec.get('error')},sort_keys=True)); return 0 if rec['status']=='tested_pass' else 1
if __name__=='__main__': raise SystemExit(main())



