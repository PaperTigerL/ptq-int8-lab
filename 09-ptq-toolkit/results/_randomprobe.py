import torch
from ptq_toolkit.pipeline import load_ptq_lab
from ptq_toolkit.models import get_model, topo_plan
from ptq_toolkit.export.topoblocks import build_blocks
lab = load_ptq_lab()
m = lab["fold_all_bn"](get_model("mobilenet_v2")).eval()
recs, blocks, stem_pool, stem_idx = build_blocks(m, topo_plan("mobilenet_v2"), False)
outs = {}
hs = [mm.register_forward_hook(lambda mod,i,o,n=n: outs.__setitem__(n,o.detach().clone())) for n,mm in m.features.named_children()]
torch.manual_seed(0); x = torch.randn(2,3,32,32)
with torch.no_grad(): m(x)
for h in hs: h.remove()
import torch.nn.functional as F
def conv(i,t):
    r=recs[i]
    y=F.conv2d(t,torch.from_numpy(r.weight),torch.from_numpy(r.bias),stride=(r.sh,r.sw),padding=(r.ph,r.pw),groups=r.groups)
    return F.relu(y) if r.relu else y
with torch.no_grad():
    y=conv(stem_idx,x)
    print("stem diff", float((y-outs["0"]).abs().max()))
    for bi,blk in enumerate(blocks):
        t=y
        for ci in blk.convs: t=conv(ci,t)
        if blk.residual:
            ident=conv(blk.ds,y) if blk.ds>=0 else y
            y=F.relu(t+ident)
        else: y=t
        key=str(bi+1) if bi<17 else "18"
        d=float((y-outs[key]).abs().max())
        print(f"block{bi:02d} res={blk.residual} diff={d:.4g} out_absmax={float(outs[key].abs().max()):.4g} min={float(outs[key].min()):.4g}")
