# Compares a qwen4_exp debug trace (ENGINE_DEBUG_TRACE_LAYERS) with the oracle dumps in results/flash_next_ref (tests/tools/ref_qwen4_exp_dump.py).
# usage: python3 -I tests/tools/qwen4_exp_trace_cmp.py <trace file>
import sys, struct
import numpy as np
trace, ref = sys.argv[1], '/home/mrq/Programs/volta-engine/results/flash_next_ref/'
H=2560
b=open(trace,'rb').read(); o=0; recs={}
while o<len(b):
    tag,m,p=struct.unpack_from('iii',b,o); o+=12
    streams = tag==1000 or 300<=tag<400 or 600<=tag<700
    cols={2000:2560,2001:1536,2002:1536,2003:1536,2004:2560}.get(tag,H)
    n=m*(4*H if streams else cols)
    a=np.frombuffer(b,dtype=np.float32,count=n,offset=o); o+=4*n
    recs[tag]=a.reshape((m,4,H) if streams else (m,cols))
def cmp(name, tag, reffile):
    if tag not in recs: print(name,'(no trace)'); return
    r=np.load(ref+reffile)[0]; a=recs[tag]; m=min(len(r),len(a)); r=r[:m]; a=a[:m]
    d=np.abs(a-r); print(f"{name:28s} maxabs {d.max():.4g} rel {np.linalg.norm(a-r)/np.linalg.norm(r):.4g} |ref|max {np.abs(r).max():.4g}")
cmp('expand streams',1000,'mod001_ExpandStreams.npy')
cmp('L0 attn mixed',100,'b2_attn_hc_mix_2.npy')
cmp('L0 attn y (after allreduce)',200,'b2_attn_out_0.npy')
cmp('L0 attn applied streams',300,'b2_attn_hc_apply_0.npy')
# GDN internals, rank 0 slice (tp=4): qkv groups [q 2048 | k 2048 | v 6144] -> rank 0 takes 512 / 512 / 1536 of each
def rec(tag): return recs.get(tag)
def rank0_qkv(a):  # full [m,10240] -> rank 0 slice [m,2560]
    return np.concatenate([a[:, 0:512], a[:, 2048:2560], a[:, 4096:5632]], axis=1)
def cmp2(name, mine, r):
    if mine is None: print(name,'(no trace)'); return
    d=np.abs(mine-r); print(f"{name:28s} maxabs {d.max():.4g} rel {np.linalg.norm(mine-r)/np.linalg.norm(r):.4g} |ref|max {np.abs(r).max():.4g}")
q=np.load(ref+'b2_attn_sub_in_proj_qkv_0.npy')[0]; z=np.load(ref+'b2_attn_sub_in_proj_z_0.npy')[0]; nrm=np.load(ref+'b2_attn_sub_norm_0.npy')[0]
cmp2('in_proj_qkv rank0 slice', rec(2000), rank0_qkv(q))
cmp2('in_proj_z rank0 slice', rec(2001), z[:, 0:1536])
cmp2('gated norm out rank0 heads', rec(2003), nrm[:, 0:12, :].reshape(len(nrm), -1))
