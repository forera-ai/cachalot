import time, numpy as np, mlx.core as mx
H,I,G=6144,3072,64
rng=np.random.default_rng(0)
N=64
ws=[mx.array(rng.integers(0,2**32,size=(I,H*3//32),dtype=np.uint32)) for _ in range(N)]
ss=[mx.array((rng.random((I,H//G))*0.01).astype(np.float32)).astype(mx.bfloat16) for _ in range(N)]
bs=[-5*s for s in ss]
mx.eval(ws,ss,bs)
for M in [1,2,3,4,6,8,12,16,24,32,48,64,128]:
    x=mx.random.normal((M,H)).astype(mx.bfloat16); mx.eval(x)
    best=1e9
    for r in range(4):
        t=time.perf_counter()
        ys=[mx.quantized_matmul(x,w,s,b,transpose=True,group_size=G,bits=3) for w,s,b in zip(ws,ss,bs)]
        mx.eval(ys); best=min(best,(time.perf_counter()-t)/N)
    print(f"M={M:4d} {best*1e6:7.1f} us  per-row {best*1e6/M:6.1f}")
