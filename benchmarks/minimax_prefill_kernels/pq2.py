import time, numpy as np, mlx.core as mx
import pq
from pq import cq, header
_MMS = r"""
  threadgroup bfloat16_t Xs[32 * 40];
  threadgroup bfloat16_t Ws[32 * 40];
  const int kp = IN_SIZE / SPLIT;
  const int k_start = threadgroup_position_in_grid.z * kp;
  const device uint8_t* wl = (const device uint8_t*)w + k_start * 3 / 8;
  qmm_t_pair_impl<bfloat16_t, 64, 3, true, 32, 32, 32>(
      (const device uint32_t*)wl, s + k_start / 64, l, x + k_start,
      y + threadgroup_position_in_grid.z * (m[0] * OUT_SIZE), Xs, Ws, IN_SIZE, OUT_SIZE, m[0], kp,
      threadgroup_position_in_grid, thread_index_in_threadgroup,
      simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""
KM = mx.fast.metal_kernel(name="minimax_pair_mms", input_names=["w","s","l","x","m"], output_names=["y"],
                          source=_MMS, header=header(), ensure_row_contiguous=True)
def split(M, out):
    if M >= 65 or (out == 6144 and M >= 33): return 1
    if out == 3072 and M <= 32: return 4
    return 2
def mms(x, w, pidx, plut):
    M, out = x.shape[0], w.shape[0]; S = split(M, out)
    y = KM(inputs=[w,pidx,plut,x,mx.array([M],dtype=mx.int32)], template=[("IN_SIZE",x.shape[-1]),("OUT_SIZE",out),("SPLIT",S)],
           grid=(128*(out//32),(M+31)//32,S), threadgroup=(128,1,1), output_shapes=[(S,M,out)], output_dtypes=[x.dtype])[0]
    return y[0] if S == 1 else y.sum(axis=0)
def pair_mm(x, w, pidx, plut):
    M = x.shape[0]
    if M == 1: return cq.qmv(x, w, pidx, None, plut)
    if M < 12: return pq.wide(x, w, pidx, plut, nv=min(M, 4))
    return mms(x, w, pidx, plut)
if __name__ == "__main__":
    rng=np.random.default_rng(3); N=24
    for out,inp in ((3072,6144),(6144,3072)):
        E=[]
        for _ in range(N):
            w=mx.array(rng.integers(0,2**32,size=(out,inp*3//32),dtype=np.uint32))
            pidx=mx.array(rng.integers(0,200,size=(out,inp//64),dtype=np.uint8))
            s=(rng.random(256)*0.01+0.001).astype(np.float32); pl=np.zeros(512,np.uint16)
            pl[0::2]=s.view(np.uint32)>>16; pl[1::2]=(-5*s).astype(np.float32).view(np.uint32)>>16
            E.append((w,pidx,mx.array(pl).view(mx.bfloat16)))
        mx.eval(E)
        bad=[]
        for M in range(1,131):
            x=mx.random.normal((M,inp)).astype(mx.bfloat16)
            w,p,l=E[0]; S,B=cq.rebuild_pair(p,l)
            if not bool(mx.array_equal(pair_mm(x,w,p,l), mx.quantized_matmul(x,w,S,B,transpose=True,group_size=64,bits=3)).item()): bad.append(M)
        print(out,inp,"mismatch at",bad,flush=True)
        for M in [1,2,4,8,11,12,16,24,32,48,64,96,128,256]:
            x=mx.random.normal((M,inp)).astype(mx.bfloat16); mx.eval(x)
            def cur():
                return [mx.quantized_matmul(x,w,*cq.rebuild_pair(p,l),transpose=True,group_size=64,bits=3) for w,p,l in E]
            def new():
                return [pair_mm(x,w,p,l) for w,p,l in E]
            r={}
            for name,f in (("cur",cur),("new",new),("cur",cur),("new",new)):
                t=time.perf_counter(); mx.eval(f()); r.setdefault(name,[]).append((time.perf_counter()-t)/N*1e6)
            print(f"  M={M:4d} cur {min(r['cur']):7.1f} us  new {min(r['new']):7.1f} us",flush=True)
