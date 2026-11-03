# Compares a qwen4_exp debug trace (debug option trace_layers) with the oracle dumps of tests/tools/ref_qwen4_exp_dump.py.
# usage: python3 -I tests/tools/qwen4_exp_trace_cmp.py <trace file> [oracle dir, default results/flash_next_ref]
import os, sys, struct
import numpy as np
trace = sys.argv[1]
ref = os.path.join( sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'flash_next_ref'), '' )
H=2560
b=open(trace,'rb').read(); o=0; recs={}
while o<len(b):
    tag,m,p=struct.unpack_from('iii',b,o); o+=12
    streams = tag==1000 or tag==800 or 300<=tag<400 or 600<=tag<700
    cols={700:20}.get(tag,H)
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
cmp('L0 mlp mixed',400,'b2_mlp_hc_mix_2.npy')
cmp('L0 moe y (after allreduce)',500,'b2_mlp_out_0.npy')
cmp('L0 mlp applied streams',600,'b2_mlp_hc_apply_0.npy')
cmp('after block 0 (mod002)',600,'mod002_TransformerBlock.npy')
cmp('after PLE (mod003)',800,'mod003_PLELayer.npy')
cmp('after block 1 (mod004)',601,'mod004_TransformerBlock.npy')
cmp('after block 2 (mod005)',602,'mod005_TransformerBlock.npy')
cmp('L3 attn mixed',103,'b6_attn_hc_mix_2.npy')
cmp('L3 attn y (after allreduce)',203,'b6_attn_out_0.npy')
cmp('L3 attn applied streams',303,'b6_attn_hc_apply_0.npy')
cmp('L3 moe y',503,'b6_mlp_out_0.npy')
cmp('after block 3 (mod006)',603,'mod006_TransformerBlock.npy')
cmp('after block 4 (mod007)',604,'mod007_TransformerBlock.npy')
cmp('after block 5 (mod008, 3-bit experts)',605,'mod008_TransformerBlock.npy')
