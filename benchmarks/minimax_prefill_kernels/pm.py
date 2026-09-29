import time, numpy as np, mlx.core as mx
import pq, pq2
from pq import cq
rng=np.random.default_rng(3); N=48
out,inp=3072,6144
E=[]
for _ in range(N):
    w=mx.array(rng.integers(0,2**32,size=(out,inp*3//32),dtype=np.uint32))
    pidx=mx.array(rng.integers(0,200,size=(out,inp//64),dtype=np.uint8))
    s=(rng.random(256)*0.01+0.001).astype(np.float32); pl=np.zeros(512,np.uint16)
    pl[0::2]=s.view(np.uint32)>>16; pl[1::2]=(-5*s).astype(np.float32).view(np.uint32)>>16
    l=mx.array(pl).view(mx.bfloat16); S,B=cq.rebuild_pair(pidx,l)
    E.append((w,pidx,l,S,B))
mx.eval(E)
def run(f,M,reps=4):
    x=mx.random.normal((M,inp)).astype(mx.bfloat16); mx.eval(x)
    mx.eval([f(x,*e) for e in E])
    best=1e9
    for _ in range(reps):
        t=time.perf_counter(); mx.eval([f(x,*e) for e in E]); best=min(best,time.perf_counter()-t)
    return best/N*1e6
pre=lambda x,w,p,l,S,B: mx.quantized_matmul(x,w,S,B,transpose=True,group_size=64,bits=3)
cur=lambda x,w,p,l,S,B: mx.quantized_matmul(x,w,*cq.rebuild_pair(p,l),transpose=True,group_size=64,bits=3)
new=lambda x,w,p,l,S,B: pq2.pair_mm(x,w,p,l)
for nv in (2,3,4,5):
    pass
for M in [1,2,3,4,6,8,11,12,16,24,32,48,64,128]:
    r=[run(f,M) for f in (pre,cur,new)]
    extra=""
    if 2<=M<12:
        extra=" wide nv: "+" ".join(f"{nv}:{run(lambda x,w,p,l,S,B: pq.wide(x,w,p,l,nv=nv),M):.1f}" for nv in (1,2,3,4,5,6,8))
    print(f"M={M:4d} pre {r[0]:6.1f} cur {r[1]:6.1f} new {r[2]:6.1f}{extra}",flush=True)
