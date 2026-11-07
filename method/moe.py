"""MoE CPU自检：直接运行本文件，不需要修改训练参数或加载数据和权重。"""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from model.model import MokioMindConfig, MoeFeedForward



def main():

    torch.manual_seed(42)
    torch.set_num_threads(1)


    cfg = MokioMindConfig(hidden_size=32, use_moe=True, dropout=0.0)
    moe = MoeFeedForward(cfg)

    x = torch.randn(2, 3, 32)

    ids, weights, aux = moe.gate(x)
    assert ids.shape == weights.shape == (6, 2)
    assert ((ids >= 0) & (ids < 4)).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(6))
    assert aux.ndim == 0 and torch.isfinite(aux) and aux.item() > 0



    with torch.no_grad():
        expected = torch.stack([
            sum(moe.experts[int(i)](token) * w for i, w in zip(indices, probs))
            for token, indices, probs in zip(x.reshape(-1, 32), ids, weights)
        ]).reshape_as(x)
        for expert in moe.shared_experts:
            expected = expected + expert(x)


    result = moe(x)
    assert result.shape == x.shape
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    (result.square().mean() + moe.aux_loss).backward()
    assert moe.gate.weight.grad is not None
    assert torch.isfinite(moe.gate.weight.grad).all()


    with torch.no_grad():

        result_eval = moe.eval()(x)
    torch.testing.assert_close(result_eval, expected, rtol=1e-5, atol=1e-6)
    assert moe.aux_loss.item() == 0
    print("PASS: MoE routing, weighted sum, router gradients, train/eval agreement")


if __name__ == "__main__":
    main()
