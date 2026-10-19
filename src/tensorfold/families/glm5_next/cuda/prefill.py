"""Packed GLM prompt helpers: bounded native micro-windows, honest plain dtype.

Plain matmul accumulates FP32 products without converting stored F16 to BF16.
Fixed per-row arithmetic; CUDA/native numerical qualification remains separate.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _plain(X,W,Y,M,xs,ys,N:tl.constexpr,K:tl.constexpr,BK:tl.constexpr,BN:tl.constexpr):
    r=tl.program_id(0)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    acc=tl.zeros((BN,),tl.float32)
    for k0 in range(0,tl.cdiv(K,BK)):
        kk=k0*BK+k
        x=tl.load(X+r*xs+kk,mask=kk<K,other=0).to(tl.float32)
        w=tl.load(W+n[:,None]*K+kk[None,:],mask=(n[:,None]<N)&(kk[None,:]<K),other=0).to(tl.float32)
        acc=acc+tl.sum(w*x[None,:],axis=1)
    tl.store(Y+r*ys+n,acc,mask=n<N)


@triton.jit
def _plain_narrow(X,W,Y,M,xs,ys,N:tl.constexpr,K:tl.constexpr,BK:tl.constexpr,BN:tl.constexpr,U:tl.constexpr):
    """``_plain``'s bytes with two outputs a warp, U of the K blocks' loads in flight a trip.

    ``_plain`` lays a 2-byte W tile [16,128] out as 8 values a thread and 16 threads across K, and so does this
    [U,2,128] tile on one warp: each K block's sum keeps its tree (an FMA chain of 8, then lane butterflies 8,4,2,1).
    The block sums then join acc in K order, each picked out of the U by a sum with zeros, which is exact; only a
    -0.0 block sum becomes +0.0, and acc (from +0.0) is never -0.0, so acc+0.0 and acc+(-0.0) agree."""
    r=tl.program_id(0)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    u=tl.arange(0,U)
    acc=tl.zeros((BN,),tl.float32)
    for k0 in range(K//(BK*U)):
        kk=(k0*U+u)[:,None,None]*BK+k[None,None,:]
        x=tl.load(X+r*xs+kk).to(tl.float32)
        w=tl.load(W+n[None,:,None]*K+kk,mask=n[None,:,None]<N,other=0).to(tl.float32)
        s=tl.sum(w*x,axis=2)
        for j in tl.static_range(U):
            acc=acc+tl.sum(tl.where(u[:,None]==j,s,0.0),axis=0)
    tl.store(Y+r*ys+n,acc,mask=n<N)


# rows a call takes on _plain_narrow (decode and verify windows); a prompt's 128-row windows keep _plain's wider tiles
NARROW_ROWS=8


def narrow_blocks(m,k,dtype):
    """K blocks a trip of _plain_narrow for an [m,k] call on a ``dtype`` W, or 0 when the call stays on _plain.

    Only 2-byte W shares _plain's 8-value thread layout; a one-block K (f_b, g_b) already runs 512 programs."""
    if m>NARROW_ROWS or k%128 or k<256 or dtype not in (torch.float16,torch.bfloat16):
        return 0
    nk=k//128
    return next(u for u in (8,4,2,1) if nk%u==0)


def plain_matmul(weight,x,out,narrow=False):
    """out = x @ weight.T in fp32; ``narrow`` (TF_GLM_PLAIN_NARROW) sends decode rows to _plain_narrow, same bytes."""
    m,k=x.shape
    n=weight.shape[0]
    if weight.shape[1]!=k or out.shape!=(m,n) or not weight.is_contiguous():
        raise ValueError('plain matmul shape/contiguity mismatch')
    u=narrow_blocks(m,k,weight.dtype) if narrow else 0
    if u:
        # GB10, one decode row, K 4096: _plain runs 2-18 programs of 16 outputs, 4 KB in flight each, at 66 GB/s
        _plain_narrow[(m,triton.cdiv(n,2))](x,weight,out,m,x.stride(0),out.stride(0),N=n,K=k,BK=128,BN=2,U=u,
                                            num_warps=1)
        return out
    _plain[(m,triton.cdiv(n,16))](x,weight,out,m,x.stride(0),out.stride(0),N=n,K=k,BK=128,BN=16,num_warps=4)
    return out


def routed_micro_windows(x,pick,ex,scratch,out,rows,limit,in_place=False):
    """Native generic compressed routed experts, copied per-slot before combine.

    Scratch slots include the shared slot (pick==E); native leaves that slot to
    caller. Shared output is installed afterward by the normal GLM combine.
    ``in_place`` (TF_GLM_MOE_IN_PLACE): each window's down epilogue writes its
    rows of ``out`` (contiguous fp32 [rows, slots, D]) instead of the scratch's
    y, which then needs no copy; the routed kernels read y nowhere else.
    """
    from tensorfold.cuda.exl3.experts import routed, ACT_BF16
    if in_place and not (out.is_contiguous() and out.dtype==torch.float32):
        raise ValueError('in-place routed windows need contiguous fp32 rows')
    own=scratch.y if in_place else None
    try:
        for start in range(0,rows,scratch.rows):
            r=min(scratch.rows,rows-start)
            if in_place:
                scratch.y=out[start:start+r].view(r*scratch.slots,ex.dims)
            y=routed(x[start:start+r],pick[start:start+r],None,ex,scratch,None,r,
                     limit=limit,act_mode=ACT_BF16)
            if y.data_ptr()!=out[start].data_ptr():
                out[start:start+r].copy_(y.view(r,scratch.slots,ex.dims))
    finally:
        if in_place:
            scratch.y=own
    return out
