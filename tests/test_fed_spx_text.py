import numpy as np
import torch
from model.compact_return import CompactReturn
from scripts.run_fed_spx_text import donors


def test_frozen_price_is_exact_after_text_learning():
    torch.manual_seed(7)
    base=CompactReturn(output_dim=2).eval()
    with torch.no_grad():
        base.price_head[-1].weight.normal_()
        base.price_head[-1].bias.normal_()
    arm=CompactReturn(text_dim=8,output_dim=2)
    arm.load_state_dict(base.state_dict(),strict=False)
    arm.freeze_price()
    x=torch.randn(5,4,35);z=torch.randn(5,8);y=torch.randn(5,2)
    expected=base(x).detach()
    assert torch.equal(arm(x,z),expected)
    original={k:v.clone() for k,v in arm.state_dict().items() if not k.startswith("text_head.")}
    opt=torch.optim.AdamW(arm.text_head.parameters(),lr=.01)
    for _ in range(3):
        opt.zero_grad();(arm(x,z)-y).square().mean().backward();opt.step()
    assert torch.equal(arm(x),expected)
    assert all(torch.equal(arm.state_dict()[k],v) for k,v in original.items())
    assert not torch.equal(arm(x,z),expected)


def test_donors_exclude_future_self_and_same_instant():
    rows=[dict(market_et=f"2020-01-0{day} 14:00:00",text_cutoff_utc=f"2020-01-0{day}T19:00:00Z") for day in (1,1,2,3)]
    a=donors(rows,23);b=donors(rows,23)
    assert np.array_equal(a,b)
    assert a[0]==a[1]==-1
    assert a[2] in (0,1)
    assert a[3] in (0,1,2)
