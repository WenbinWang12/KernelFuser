from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("triton.language.extra.tlx")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swiglu import native_swiglu_tlx, native_swiglu_triton


@pytest.mark.parametrize("backend", [native_swiglu_tlx, native_swiglu_triton])
@pytest.mark.parametrize("shape", [(1025,), (17, 65)])
def test_swiglu_forward_and_backward_match_torch(backend, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if backend is native_swiglu_tlx and torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("native TLX backend requires Hopper/SM90")

    if shape == (17, 65):
        gate = torch.randn((65, 17), device="cuda", dtype=torch.bfloat16).t()
        up = torch.randn((65, 17), device="cuda", dtype=torch.bfloat16).t()
        grad = torch.randn((65, 17), device="cuda", dtype=torch.bfloat16).t()
    else:
        gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        grad = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    gate_ref = gate.detach().requires_grad_()
    up_ref = up.detach().requires_grad_()
    output_ref = torch.nn.functional.silu(gate_ref) * up_ref
    grad_gate_ref, grad_up_ref = torch.autograd.grad(output_ref, (gate_ref, up_ref), grad)

    output = backend.forward(gate, up)
    grad_gate, grad_up = backend.backward(gate, up, grad)

    torch.testing.assert_close(output, output_ref, atol=0.08, rtol=0.02)
    torch.testing.assert_close(grad_gate, grad_gate_ref, atol=0.08, rtol=0.02)
    torch.testing.assert_close(grad_up, grad_up_ref, atol=0.08, rtol=0.02)
