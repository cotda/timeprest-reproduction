"""The graph backward rule (swap.GraphForward) on small cases with known answers."""
import copy

import pytest
import torch
import torch.nn as nn
from torch.func import functional_call

from timeprest.swap import GraphForward

D = torch.float64


def test_scalar_example_matches_closed_form():
    """paper_notes §14.4: a = w1*x, y = w2*a, loss = (y - t)^2, forward at old weights, backward at
    new ones: g_w2 = r*a_old, g_w1 = r*w2_new*x with r = 2*(y_old - t)."""
    w1o, w2o, w1n, w2n, x, t = (torch.tensor(v, dtype=D) for v in (0.7, -1.3, 0.4, 2.1, 1.5, 0.2))
    P = {"w1": w1o.clone().requires_grad_(), "w2": w2o.clone().requires_grad_()}
    gf = GraphForward(P)
    loss = gf.run(lambda: (P["w2"] * (P["w1"] * x) - t) ** 2)
    gf.use({"w1": w1n, "w2": w2n})
    g1, g2 = torch.autograd.grad(loss, [P["w1"], P["w2"]])
    r = 2 * (w2o * w1o * x - t)
    torch.testing.assert_close(g2, r * w1o * x)
    torch.testing.assert_close(g1, r * w2n * x)


@pytest.mark.parametrize("layer", ["linear", "conv_bn"])
def test_matches_in_place_copy_before_backward(layer):
    """Same gradients as PipeDream's mechanism (copy the backward version into the module in place,
    then backward on the stored graph), including weights saved as views (Linear saves weight.t())
    and BN (batch statistics of the forward, new scale)."""
    torch.manual_seed(0)
    m = (nn.Sequential(nn.Linear(6, 5), nn.Tanh(), nn.Linear(5, 3)) if layer == "linear" else
         nn.Sequential(nn.Conv2d(2, 4, 3), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(),
                       nn.Linear(4 * 4 * 4, 3))).double().train()
    x = torch.randn(4, 6, dtype=D) if layer == "linear" else torch.randn(4, 2, 6, 6, dtype=D)
    old = {n: p.detach().clone() for n, p in m.named_parameters()}
    new = {n: p + 0.3 * torch.randn_like(p) for n, p in old.items()}

    ref = copy.deepcopy(m)
    xr = x.clone().requires_grad_(True)
    out_r = ref(xr)
    for n, p in ref.named_parameters():
        p.data.copy_(new[n])
    (out_r ** 2).sum().backward()

    P = {n: t.clone().requires_grad_(True) for n, t in old.items()}
    xg = x.clone().requires_grad_(True)
    gf = GraphForward(P)
    mod = copy.deepcopy(m)
    out = gf.run(lambda: functional_call(mod, P, (xg,)))
    gf.use(new)
    g = torch.autograd.grad((out ** 2).sum(), list(P.values()) + [xg])
    for (n, p), gp in zip(ref.named_parameters(), g):
        torch.testing.assert_close(gp, p.grad, msg=n)
    torch.testing.assert_close(g[-1], xr.grad)
    # and it is not simply the forward-version gradient
    P2 = {n: t.clone().requires_grad_(True) for n, t in old.items()}
    g_old = torch.autograd.grad((functional_call(copy.deepcopy(m), P2, (x,)) ** 2).sum(), list(P2.values()))
    assert any(not torch.allclose(a, b) for a, b in zip(g, g_old))


def test_backward_before_use_is_an_error():
    P = {"w": torch.ones(2, 2, dtype=D, requires_grad=True)}
    gf = GraphForward(P)
    x = torch.ones(3, 2, dtype=D, requires_grad=True)
    out = gf.run(lambda: x @ P["w"])
    with pytest.raises(RuntimeError):
        torch.autograd.grad(out.sum(), [x])
