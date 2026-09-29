import os,sys,time,numpy as np,mlx.core as mx
import importlib.util
os.environ["NS"]="1"; os.environ["LAYERS"]="1"
spec=importlib.util.spec_from_file_location("rb","rb_price.py"); rb=importlib.util.module_from_spec(spec); spec.loader.exec_module(rb)
import pq2
def q(x,t,arm,proj):
    if arm=="mix": arm = "new" if proj!="w2" else "cur"
    if arm=="w2c":  # w2 through pair kernel but input made contiguous via explicit copy
        arm="new"
    return rb.qmm(x,t,arm)
def lay(x,M,arm):
    outs=[]
    for e in range(128):
        xe=x[(e*M)%(x.shape[0]-M):(e*M)%(x.shape[0]-M)+M]
        t=rb.experts[e]
        g=q(xe,t["w1"],arm,"w1"); u=q(xe,t["w3"],arm,"w3")
        h=mx.sigmoid(g)*g*u
        if arm=="w2c": mx.async_eval(h)
        outs.append(q(h,t["w2"],arm,"w2"))
    y=mx.concatenate(outs,axis=0); mx.eval(y)
x=mx.random.normal((4096,6144)).astype(mx.bfloat16)
for M in [4,16,48,128]:
    r={}
    for arm in ["cur","new","mix","pre"]*3:
        t=time.perf_counter(); [lay(x,M,arm) for _ in range(3)]; r.setdefault(arm,[]).append(time.perf_counter()-t)
    print(f"M={M} "+" ".join(f"{a}={min(v)*1000:.1f}ms" for a,v in r.items()),flush=True)
