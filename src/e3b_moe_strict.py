#!/usr/bin/env python3
"""Development implementation of graphable, padded complete real MoE layers.

Candidate stages RMS statistics once and applies post-attention norm inside
the routed experts' gate/up GEMMs. Routing remains the model's native operation.
Every expert has capacity T (padding is masked); this schedule is explicitly
reported and is NOT claimed to be a production sparse dispatch implementation.
"""
import argparse, gc, gzip, hashlib, json, math, os, random, statistics, subprocess, time, traceback, platform, sys, threading
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from safetensors import safe_open

ATOL, RTOL, LR = 0.02, 0.02, 0.01

@triton.jit
def _stats(X,R,K:tl.constexpr,EPS:tl.constexpr,B:tl.constexpr):
    m=tl.program_id(0); k=tl.arange(0,B)
    x=tl.load(X+m*K+k,k<K,0.).to(tl.float32)
    tl.store(R+m,tl.rsqrt(tl.sum(x*x,0)/K+EPS))

@triton.jit
def _norm_gu(X,G,R,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); kk=tl.arange(0,BK)
    rr=tl.load(R+m,m<M,0.); acc=tl.zeros((BM,BN),tl.float32)
    for b in range(tl.cdiv(K,BK)):
        k=b*BK+kk
        x=tl.load(X+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),0.).to(tl.float32)
        g=tl.load(G+k,k<K,0.).to(tl.float32)
        # Keep both BF16 roundings of the HF RMSNorm implementation.
        z=(x*rr[:,None]).to(tl.bfloat16)
        z=(z.to(tl.float32)*g[None,:]).to(tl.bfloat16)
        w=tl.load(W+n[None,:]*K+k[:,None],(n[None,:]<N)&(k[:,None]<K),0.)
        acc=tl.dot(z,w,acc)
    tl.store(Y+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))

def norm(x,g,eps):
    u=x.float(); return (u*torch.rsqrt(u.square().mean(-1,keepdim=True)+eps)).to(x.dtype)*g

class ExpertProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,g,r,w,eps):
        m,k=x.shape;n=w.shape[0];out=torch.empty((m,n),device=x.device,dtype=x.dtype)
        _norm_gu[(triton.cdiv(m,32),triton.cdiv(n,64))](x,g,r,w,out,m,n,k,32,64,32,num_warps=4)
        ctx.save_for_backward(x,g,w);ctx.eps=eps
        return out
    @staticmethod
    def backward(ctx,dy):
        x,g,w=ctx.saved_tensors
        # Independent native BF16 derivative, preserving the norm's casts.
        with torch.enable_grad():
            xx=x.detach().requires_grad_(True);gg=g.detach().requires_grad_(True);ww=w.detach().requires_grad_(True)
            y=F.linear(norm(xx,gg,ctx.eps),ww)
            gx,ggg,gw=torch.autograd.grad(y,(xx,gg,ww),dy)
        return gx,ggg,None,gw,None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(a,b):
    a,b=a.detach().float(),b.detach().float();d=a-b
    return {'allclose':bool(torch.allclose(a,b,atol=ATOL,rtol=RTOL)),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'max_abs':float(d.abs().max()),'relative_l2':float(d.norm()/b.norm().clamp_min(1e-8))}
def aggregate(v):
    sem=statistics.stdev(v)/math.sqrt(len(v)) if len(v)>1 else 0
    return {'samples_ms':v,'mean_ms':statistics.fmean(v),'median_ms':statistics.median(v),'sem_ms':sem,'two_sem_ms':2*sem,'n':len(v)}

def load(root):
    c=json.loads((root/'config.json').read_text());kind=c['model_type']
    assert kind in ('olmoe','qwen2_moe','deepseek_v2'),kind
    E=c.get('num_experts',c.get('n_routed_experts')); layer=1 if kind=='deepseek_v2' else 0
    ix=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']; files={};sources={}
    def get(k):
        f=root/ix[k]
        with safe_open(str(f),framework='pt',device='cpu') as z:v=z.get_tensor(k)
        sources[k]={'shard':ix[k],'shape':list(v.shape),'dtype':str(v.dtype)}
        return v.to('cuda',torch.bfloat16)
    pref=f'model.layers.{layer}.';p={'gamma':get(pref+'post_attention_layernorm.weight'),'router':get(pref+'mlp.gate.weight')}
    for e in range(E):
        pre=pref+f'mlp.experts.{e}.'
        p[f'gu_{e}']=torch.cat([get(pre+'gate_proj.weight'),get(pre+'up_proj.weight')],0)
        p[f'down_{e}']=get(pre+'down_proj.weight')
    if kind in ('qwen2_moe','deepseek_v2'):
        pre=pref+('mlp.shared_expert.' if kind=='qwen2_moe' else 'mlp.shared_experts.')
        for k in ['gate_proj','up_proj','down_proj']:p['shared_'+k]=get(pre+k+'.weight')
        if kind=='qwen2_moe':p['shared_gate']=get(pref+'mlp.shared_expert_gate.weight')
    c.update(E=E,layer=layer,eps=c.get('rms_norm_eps',1e-6))
    return c,p,sources

def router(z,p,c):
    # HF OLMoE/Qwen2 use model dtype for linear logits, FP32 softmax.
    # DeepSeek-V2's public gate casts its linear inputs and weights to FP32.
    if c['model_type']=='deepseek_v2':logits=F.linear(z.float(),p['router'].float())
    else:logits=F.linear(z,p['router'])
    s=F.softmax(logits,dim=-1,dtype=torch.float32);top=c['num_experts_per_tok']
    if c['model_type']=='deepseek_v2':
        ng=c['n_group'];gs=s.view(-1,ng,c['E']//ng).max(-1).values
        gi=gs.topk(c['topk_group'],dim=-1,sorted=False).indices
        mask=torch.zeros_like(gs).scatter_(1,gi,1).unsqueeze(-1).expand(-1,-1,c['E']//ng).reshape_as(s)
        wt,ids=s.masked_fill(~mask.bool(),0).topk(top,dim=-1,sorted=False)
    else:wt,ids=s.topk(top,dim=-1)
    if c.get('norm_topk_prob',False):wt=wt/wt.sum(-1,keepdim=True)
    elif c['model_type']=='deepseek_v2':wt=wt*c['routed_scaling_factor']
    return ids,wt.to(z.dtype),logits

DISPATCH = None

def make_dispatch(ids, c):
    """Materialize a static expert dispatch schedule for CUDA Graph replay."""
    chunks=[]
    for e in range(c['E']):
        ps=[]; ss=[]
        for sl in range(c['num_experts_per_tok']):
            pos=(ids[:, sl] == e).nonzero(as_tuple=False).flatten()
            ps.append(pos); ss.append(torch.full_like(pos, sl))
        pos=torch.cat(ps) if ps else ids.new_empty((0,))
        slot=torch.cat(ss) if ss else ids.new_empty((0,))
        chunks.append((pos,slot))
    cap=max((q[0].numel() for q in chunks), default=1)
    out=[]
    for pos,slot in chunks:
        pad=cap-pos.numel()
        out.append((F.pad(pos,(0,pad)), F.pad(slot,(0,pad)),
                    torch.cat([torch.ones(pos.numel(),device=ids.device,dtype=torch.bool),
                               torch.zeros(pad,device=ids.device,dtype=torch.bool)])))
    return out

def module(x,p,c,candidate=False,sparse=False):
    z=norm(x,p['gamma'],c['eps']);ids,wt,logits=router(z,p,c)
    if candidate:
        r=torch.empty(x.shape[0],device=x.device,dtype=torch.float32)
        _stats[(x.shape[0],)](x,r,x.shape[1],c['eps'],triton.next_power_of_2(x.shape[1]),num_warps=8)
    out=torch.zeros_like(x)
    dispatch = DISPATCH if DISPATCH is not None else make_dispatch(ids,c)
    for e in range(c['E']):
        w=p[f'gu_{e}']; n=w.shape[0]//2
        pos,slot,active=dispatch[e]
        if sparse:
            # Independent native sparse reference. Never captured/timed.
            slot,pos=torch.where((ids==e).T)
            if pos.numel()==0:continue
            zz=z[pos];gate=F.linear(zz,w[:n]);up=F.linear(zz,w[n:]);v=F.linear(F.silu(gate)*up,p[f'down_{e}'])
            out.index_add_(0,pos,v*wt[pos,slot,None])
        else:
            xe=x.index_select(0,pos) if candidate else z.index_select(0,pos)
            if candidate:gu=ExpertProjection.apply(xe,p['gamma'],r.index_select(0,pos),w,c['eps']);gate,up=gu.split(n,-1)
            else:gate=F.linear(xe,w[:n]);up=F.linear(xe,w[n:])
            v=F.linear(F.silu(gate)*up,p[f'down_{e}'])
            ew=wt.index_select(0,pos).gather(1,slot[:,None]).squeeze(1).to(v.dtype)*active.to(v.dtype)
            out.index_add_(0,pos,v*ew[:,None])
    if c['model_type'] in ('qwen2_moe','deepseek_v2'):
        sh=F.linear(F.silu(F.linear(z,p['shared_gate_proj']))*F.linear(z,p['shared_up_proj']),p['shared_down_proj'])
        if c['model_type']=='qwen2_moe':sh=sh*torch.sigmoid(F.linear(z,p['shared_gate']))
        out=out+sh
    return out,ids,wt,logits

def clone(p,x,dtype=None):
    typ=dtype or x.dtype
    return x.detach().to(typ).clone().requires_grad_(True),{k:v.detach().to(typ).clone().requires_grad_(True) for k,v in p.items()}
def grad_run(x,p,c,candidate,sparse=False):
    y,ids,wt,logits=module(x,p,c,candidate,sparse)
    # Fixed cotangent avoids tiny y.square().mean gradients that can make a
    # BF16 update silently round to zero. Same cotangent in every arm.
    dy=torch.sin(torch.arange(y.numel(),device=y.device,dtype=torch.float32)).reshape_as(y)/math.sqrt(y.numel())
    gs=torch.autograd.grad((y.float()*dy).sum(),[x]+list(p.values()),allow_unused=True)
    return y,ids,wt,gs

def correctness(x,p,c):
    # BF16 native sparse module vs padded baseline vs staged candidate.
    xx,pp=clone(p,x);ys,is_,ws,gs=grad_run(xx,pp,c,False,True)
    ref={'y':ys.detach(),'ids':is_.detach(),'wt':ws.detach(),'grads':[None if q is None else q.detach() for q in gs]}
    del xx,pp,ys,is_,ws,gs;gc.collect();torch.cuda.empty_cache()
    results={}
    for name,cand in [('baseline',False),('staged',True)]:
        xx,pp=clone(p,x);y,ids,wt,gs=grad_run(xx,pp,c,cand)
        gm=[];updates=[];changed=0
        for k,v,g,h in zip(['input']+list(pp),[xx]+list(pp.values()),gs,ref['grads']):
            if h is None:h=torch.zeros_like(v)
            if g is None:g=torch.zeros_like(v)
            gm.append({'parameter':k,**metric(g,h)})
            # FP32 master SGD update, cast back to BF16 for next step.
            before=v.detach();u=(before.float()-LR*g.float()).to(v.dtype);ur=(before.float()-LR*h.float()).to(v.dtype)
            ch=int((u!=before).count_nonzero());changed+=ch
            updates.append({'parameter':k,'changed':ch,**metric(u,ur)})
        results[name]={'forward':metric(y,ref['y']),'routing_equal':bool(torch.equal(ids,ref['ids'])),'routing_weights':metric(wt,ref['wt']),'gradients':gm,'updates':updates,'changed_parameter_elements':changed}
        del xx,pp,y,ids,wt,gs;gc.collect();torch.cuda.empty_cache()
    # Independent FP32 module reference uses same checkpoint weights and all
    # routes computed afresh. Route ties can legitimately differ in BF16;
    # require equality here to qualify this configuration.
    xx,pp=clone(p,x,torch.float32);yf,ifi,wf,gf=grad_run(xx,pp,c,False,True)
    results['fp32_oracle']={'bf16_forward':metric(ref['y'],yf),'routing_equal':bool(torch.equal(ref['ids'],ifi))}
    for name in ('baseline','staged'):
        # Separate BF16-vs-native gradient check above; oracle numerical
        # gradient checks retain explicit parameter names.
        results[name]['fp32_oracle_gradients']=[]
    # Hold only the FP32 grads while re-evaluating each BF16 arm serially.
    fpgr=[None if q is None else q.detach() for q in gf];del xx,pp,yf,ifi,wf,gf
    for name,cand in [('baseline',False),('staged',True)]:
        xx,pp=clone(p,x);_,_,_,gs=grad_run(xx,pp,c,cand)
        for k,v,g,h in zip(['input']+list(pp),[xx]+list(pp.values()),gs,fpgr):
            if g is None:g=torch.zeros_like(v)
            if h is None:h=torch.zeros_like(v,dtype=torch.float32)
            results[name]['fp32_oracle_gradients'].append({'parameter':k,**metric(g,h)})
        del xx,pp,gs;gc.collect();torch.cuda.empty_cache()
    checks=[]
    for name in ('baseline','staged'):
        q=results[name];checks.extend([q['forward']['allclose'],q['routing_equal'],q['changed_parameter_elements']>0]);checks.extend(v['allclose'] and v['finite'] for group in ('gradients','updates','fp32_oracle_gradients') for v in q[group])
    checks.extend([results['fp32_oracle']['bf16_forward']['allclose']])
    results['pass']=all(checks);return results

def capture(x,p,c,cand,mode):
    xx,pp=clone(p,x);keep=[xx,pp];stream=torch.cuda.Stream();g=torch.cuda.CUDAGraph()
    def fn():
        if mode=='forward':return module(xx,pp,c,cand)[0]
        y,ids,wt,gr=grad_run(xx,pp,c,cand)
        if mode=='step':
            with torch.no_grad():
                # Update is written to separate destinations. Each replay
                # benchmarks the same training step and routing input.
                for u,v,q in zip(update_out,[xx]+list(pp.values()),gr):
                    if q is None:u.copy_(v)
                    else:torch.add(v.float(),q.float(),alpha=-LR,out=u)
        return y
    update_out=[torch.empty_like(v,dtype=torch.float32) for v in [xx]+list(pp.values())] if mode=='step' else []
    keep.append(update_out)
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    stream.synchronize()
    with torch.cuda.graph(g,stream=stream):out=fn()
    keep.append(out);return g,keep

def timing(x,p,c,mode,warmup,samples,rounds,seed):
    vals={'baseline':[],'staged':[]};orders=[]
    for rd in range(rounds):
        graphs={name:capture(x,p,c,cand,mode) for name,cand in [('baseline',False),('staged',True)]}
        for g,_ in graphs.values():
            for _ in range(warmup):g.replay()
        torch.cuda.synchronize();rng=random.Random(seed+rd)
        for i in range(samples):
            order=list(graphs);rng.shuffle(order);orders.append(order)
            for name in order:
                st,en=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);st.record();graphs[name][0].replay();en.record();en.synchronize();vals[name].append(st.elapsed_time(en))
        del graphs;gc.collect();torch.cuda.empty_cache()
    ans={k:aggregate(v) for k,v in vals.items()};ans['speedup']=ans['baseline']['mean_ms']/ans['staged']['mean_ms'];ans['orders']=orders;return ans

def trace(x,p,c,cand,mode,path):
    g,keep=capture(x,p,c,cand,mode)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as pr:
        g.replay();torch.cuda.synchronize()
    raw=path.with_suffix('');pr.export_chrome_trace(str(raw))
    with raw.open('rb') as fi,gzip.open(path,'wb') as fo:fo.write(fi.read())
    raw.unlink();return str(path)

def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _gpu_snapshot(physical_gpu):
    """Return physical-GPU identity and all compute PIDs at one instant."""
    rec = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'physical_gpu': str(physical_gpu), 'pid': os.getpid(),
           'query_ok': False, 'exclusive': False, 'apps': [], 'foreign_apps': []}
    try:
        q = subprocess.run(['nvidia-smi', '-i', str(physical_gpu),
                            '--query-gpu=index,uuid,name', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, check=True, timeout=10)
        line = next((x.strip() for x in q.stdout.splitlines() if x.strip()), '')
        fields = [x.strip() for x in line.split(',')]
        if len(fields) >= 3:
            rec['gpu_index'], rec['gpu_uuid'], rec['gpu_name'] = fields[:3]
        a = subprocess.run(['nvidia-smi', '-i', str(physical_gpu),
                            '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                            '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, check=True, timeout=10)
        for line in a.stdout.splitlines():
            fields = [x.strip() for x in line.split(',')]
            if len(fields) < 2 or not fields[1].isdigit():
                continue
            app = {'gpu_uuid': fields[0], 'pid': int(fields[1]),
                   'process_name': fields[2] if len(fields) > 2 else '',
                   'used_memory_mib': fields[3] if len(fields) > 3 else ''}
            rec['apps'].append(app)
            if app['pid'] != os.getpid():
                rec['foreign_apps'].append(app)
        rec['query_ok'] = True
        rec['exclusive'] = len(rec['foreign_apps']) == 0
    except Exception as exc:
        rec['error'] = repr(exc)
    return rec


class _ExclusivityWindow:
    def __init__(self, physical_gpu):
        self.physical_gpu = str(physical_gpu)
        self.records = []
        self.stop = threading.Event()
        self.thread = None
        self.started = 0.0

    def start(self):
        self.started = time.monotonic()
        self.records.append(_gpu_snapshot(self.physical_gpu))
        def poll():
            while not self.stop.wait(1.0):
                self.records.append(_gpu_snapshot(self.physical_gpu))
        self.thread = threading.Thread(target=poll, name='e3b-gpu-exclusivity', daemon=True)
        self.thread.start()

    def finish(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=15.0)
        self.records.append(_gpu_snapshot(self.physical_gpu))
        return {'poll_interval_seconds': 1.0,
                'window_seconds': time.monotonic() - self.started,
                'observations': self.records,
                'passed': bool(self.records) and all(x.get('query_ok') and x.get('exclusive') for x in self.records)}


def _dispatch_audit(ids, c, dispatch):
    """Audit that the static capacity-padded schedule covers native routes exactly."""
    expected = []
    experts = []
    for e in range(c['E']):
        pairs = []
        for sl in range(c['num_experts_per_tok']):
            pos = (ids[:, sl] == e).nonzero(as_tuple=False).flatten()
            pairs.extend((int(p), int(sl)) for p in pos.detach().cpu().tolist())
        expected.extend((e, p, sl) for p, sl in pairs)
        pos, slot, active = dispatch[e]
        active_pos = pos[active].detach().cpu().tolist()
        active_slot = slot[active].detach().cpu().tolist()
        got = list(zip(active_pos, active_slot))
        experts.append({'expert': e, 'active': len(got), 'capacity': int(pos.numel()),
                        'padding': int((~active).sum().item()),
                        'active_pairs_sha256': hashlib.sha256(json.dumps(got, separators=(',', ':')).encode()).hexdigest(),
                        'coverage_exact': sorted(got) == sorted(pairs),
                        'unique_pairs': len(set(got)) == len(got)})
    route_pairs = sorted(expected)
    padded_pairs = sorted((e, int(p), int(s)) for e, item in enumerate(dispatch)
                          for p, s in zip(item[0][item[2]].detach().cpu().tolist(),
                                          item[1][item[2]].detach().cpu().tolist()))
    return {'tokens': int(ids.shape[0]), 'top_k': int(c['num_experts_per_tok']),
            'n_experts': int(c['E']), 'capacity': max(x['capacity'] for x in experts),
            'active_assignments': len(route_pairs),
            'padded_slots': sum(x['capacity'] for x in experts),
            'padding_slots': sum(x['padding'] for x in experts),
            'route_pairs_sha256': hashlib.sha256(json.dumps(route_pairs, separators=(',', ':')).encode()).hexdigest(),
            'coverage_exact': route_pairs == padded_pairs and all(x['coverage_exact'] and x['unique_pairs'] for x in experts),
            'experts': experts}


def _trace_manifest(paths):
    out = []
    for path in paths:
        p = Path(path)
        item = {'path': str(p), 'exists': p.is_file()}
        if p.is_file():
            item['size_bytes'] = p.stat().st_size
            item['sha256'] = _sha256(p)
        out.append(item)
    return out


def _run_one(model_path, out, trace_dir, physical_gpu, warmup, samples, rounds, tokens):
    global DISPATCH
    c, p, sources = load(Path(model_path))
    x = torch.randn((tokens, c['hidden_size']), device='cuda', dtype=torch.bfloat16)
    with torch.no_grad():
        z = norm(x, p['gamma'], c['eps']); ids, _, _ = router(z, p, c)
        DISPATCH = make_dispatch(ids, c)
    dispatch_audit = _dispatch_audit(ids, c, DISPATCH)
    if not dispatch_audit['coverage_exact']:
        raise RuntimeError('native_padded_dispatch_coverage_failed')
    row = {'model': c['model_type'], 'tokens': tokens, 'status': 'running',
           'correctness': None, 'dispatch_audit': dispatch_audit, 'traces': []}
    row['correctness'] = correctness(x, p, c)
    if not row['correctness'].get('pass'):
        raise RuntimeError('correctness_gate_failed')
    for mode in ('forward', 'backward', 'step'):
        row[mode] = timing(x, p, c, mode, warmup, samples, rounds, 4101)
        if row[mode]['speedup'] > 1.01:
            row[mode + '_retest'] = timing(x, p, c, mode, warmup, samples, rounds, 5132)
        for name, cand in (('baseline', False), ('staged', True)):
            path = trace_dir / f'{c["model_type"]}_{name}_{mode}.chrome_trace.json.gz'
            trace(x, p, c, cand, mode, path)
            row['traces'].append(str(path))
    row['trace_manifest'] = _trace_manifest(row['traces'])
    row['status'] = 'tested_pass'
    return row, c, sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--trace-dir', required=True, type=Path)
    ap.add_argument('--physical-gpu', required=True)
    ap.add_argument('--tokens', type=int, default=32)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--rounds', type=int, default=5)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.trace_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        'schema': 'kernelfuser-e3b-strict-audited-v1', 'status': 'running',
        'publication_eligible': False, 'command': sys.argv,
        'source': {'path': str(Path(__file__).resolve()), 'sha256': _sha256(Path(__file__))},
        'environment': {'host': platform.node(), 'pid': os.getpid(),
                        'python': sys.version, 'python_executable': sys.executable,
                        'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES', ''),
                        'physical_gpu': str(a.physical_gpu), 'gpu': torch.cuda.get_device_name()},
        'protocol': {'warmup': a.warmup, 'samples': a.samples, 'rounds': a.rounds,
                     'cuda_graph': True, 'paired_interleaved': True,
                     'correctness_before_timing': True, 'forward_backward_step': True,
                     'retest_trigger': '>1.01x staged speedup',
                     'gpu_exclusivity': 'physical GPU sampled before/after and every second; foreign PID invalidates run',
                     'native_scope': 'real checkpoint router, capacity-padded expert dispatch; native sparse reference audited'},
        'scope': {'level': 'one real checkpoint MoE layer',
                  'claim_boundary': 'module-level forward/backward/update; no transformer E2E claim'},
        'physical_gpu': str(a.physical_gpu), 'rows': [], 'reasons': []}
    window = _ExclusivityWindow(a.physical_gpu)
    try:
        pre = _gpu_snapshot(a.physical_gpu); receipt['exclusivity_pre'] = pre
        if not pre.get('exclusive'):
            raise RuntimeError('gpu_not_exclusive_at_start')
        window.start()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(20261013)
        row, config, sources = _run_one(a.model_path, a.out, a.trace_dir,
                                         a.physical_gpu, a.warmup, a.samples, a.rounds, a.tokens)
        receipt['checkpoint'] = {'path': a.model_path, 'config': config,
                                 'config_sha256': _sha256(Path(a.model_path) / 'config.json'),
                                 'tensors': sources}
        receipt['rows'].append(row)
        receipt['status'] = 'tested_pass' if (a.warmup, a.samples, a.rounds) == (20, 100, 5) else 'development_pass'
        receipt['publication_eligible'] = receipt['status'] == 'tested_pass'
    except Exception as exc:
        receipt.update({'status': 'tested_invalid', 'error': repr(exc),
                        'traceback': traceback.format_exc()})
        receipt['reasons'].append(type(exc).__name__)
    finally:
        receipt['exclusivity'] = window.finish() if window.started else {'passed': False, 'observations': []}
        receipt['exclusivity_post'] = _gpu_snapshot(a.physical_gpu)
        if not receipt['exclusivity'].get('passed') or not receipt['exclusivity_post'].get('exclusive'):
            receipt['publication_eligible'] = False
            receipt['status'] = 'tested_invalid'
            receipt['reasons'].append('gpu_exclusivity_gate_failed')
        for row in receipt.get('rows', []):
            row['trace_manifest'] = _trace_manifest(row.get('traces', []))
        receipt['trace_manifest'] = [item for row in receipt.get('rows', []) for item in row.get('trace_manifest', [])]
        receipt['trace_count'] = len(receipt['trace_manifest'])
        receipt['trace_all_present'] = receipt['trace_count'] == 6 and all(x.get('exists') and x.get('sha256') for x in receipt['trace_manifest'])
        if not receipt['trace_all_present']:
            receipt['publication_eligible'] = False
            receipt['status'] = 'tested_invalid'
            receipt['reasons'].append('trace_manifest_gate_failed')
        a.out.write_text(json.dumps(receipt, indent=2, default=str) + '\n')
    print(json.dumps({'status': receipt['status'], 'publication_eligible': receipt['publication_eligible'],
                      'trace_count': receipt.get('trace_count'), 'out': str(a.out)}, indent=2), flush=True)
    return 0 if receipt['publication_eligible'] else 1


if __name__ == '__main__':
    raise SystemExit(main())


