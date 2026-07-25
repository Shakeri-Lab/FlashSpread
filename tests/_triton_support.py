"""Capability probes for the optional Triton stack, shared by the test suite.

Some tests get device-free coverage of real Triton kernels by running them under
``TRITON_INTERPRET=1``. That interpreter is not uniformly capable across the
Triton versions this package declares support for, and its failure modes are
misleading in both directions:

* it can reject valid kernels (Triton 3.1 lacks ``InterpreterBuilder.get_int1_ty``,
  and resolves ``constexpr`` parameters with an exact ``== "constexpr"`` string
  comparison that a stringified annotation defeats);
* it can also *succeed* while its stores never reach the host tensors, which
  turns every downstream assertion into a tautology.

Probing both properties directly is more durable than pinning a version range,
and it keeps a genuine package regression distinguishable from an interpreter
limitation.
"""

import functools
import os
from pathlib import Path
import subprocess
import sys


_PROBE = """
import torch
import triton
import triton.language as tl


@triton.jit
def probe(out_ptr, flag_ptr, N, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    # tl.where over a boolean exercises the int1 type the interpreter builder
    # must provide; the stores then prove results reach host storage at all.
    value = tl.where(mask, idx.to(tl.int32) + 1, 0)
    tl.store(out_ptr + idx, value, mask=mask)
    tl.store(flag_ptr + idx, mask.to(tl.int32), mask=mask)


out = torch.zeros(8, dtype=torch.int32)
flag = torch.zeros(8, dtype=torch.int32)
probe[(1,)](out, flag, 8, BLOCK=8)
expected = torch.arange(1, 9, dtype=torch.int32)
if not torch.equal(out, expected):
    raise SystemExit(f"interpreter stores did not reach host storage: {out.tolist()}")
if int(flag.sum()) != 8:
    raise SystemExit("interpreter mask stores did not reach host storage")
"""


@functools.lru_cache(maxsize=1)
def triton_interpreter_skip_reason() -> str | None:
    """Return ``None`` when the Triton interpreter can run FlashSpread kernels.

    Otherwise return a reason naming the installed Triton version. The probe
    runs in a subprocess so ``TRITON_INTERPRET`` never leaks into the parent,
    which must stay able to launch real CUDA kernels.
    """
    try:
        import triton
    except Exception:  # pragma: no cover - Triton is an optional dependency
        return "Triton is an optional GPU dependency and is not installed"

    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
    except subprocess.SubprocessError as exc:  # pragma: no cover - probe failure
        return f"could not probe the Triton {triton.__version__} interpreter: {exc}"

    if result.returncode == 0:
        return None
    detail = (result.stdout + result.stderr).strip().splitlines()
    summary = detail[-1] if detail else "unknown failure"
    return (
        f"the Triton {triton.__version__} interpreter cannot execute these "
        f"kernels ({summary}); the compiled path is covered by the gpu-marked "
        "tests instead"
    )
