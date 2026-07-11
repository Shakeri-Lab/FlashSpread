#!/usr/bin/env python
"""
Age-dependent shedding: naive dense pre-pass vs fused in-kernel infectivity.

Reproduces Table `tab:agedep_overhead`. Two implementations are compared, each
under both transmission modes (constant beta vs age-dependent s(tau)):

  * In-kernel  -- the production fused CUDA-Graph engine, which writes the
                  next-step infectivity inside the fused Triton kernel's tail
                  (only lanes that will be I next step pay the erfcx cost).

  * Pre-pass   -- the naive alternative: the fused kernel is built in *constant*
                  mode (so it does no in-kernel hazard work) and a dense PyTorch
                  pass over ALL N nodes computes the infectivity before every
                  step (one torch erfcx per node, masked to I afterwards). This
                  is implemented here as a benchmark-local subclass that prepends
                  the dense pass to the captured step -- the shipped engine is
                  NOT modified.

Slowdown is reported as constant/age_dependent - 1 within each implementation,
so it isolates the cost of the age-dependent hazard under each design.

Both use the `thread` CSR strategy to match the manuscript's harness.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread import SEIRModel  # noqa: E402
from flashspread.core.network import BarabasiAlbertGraph, FixedDegreeGraph  # noqa: E402
from flashspread.engines.renewal_fused import RenewalEngineFusedCUDAGraph  # noqa: E402
from flashspread.models.hazards import lognormal_hazard_stable  # noqa: E402

BETA = 0.25
EPSILON = 0.03
TAU_MAX = 0.1
STEPS_PER_LAUNCH = 50
TF = 50.0


class NaivePrepassFusedCG(RenewalEngineFusedCUDAGraph):
    """Fused CG engine with a naive dense infectivity pre-pass prepended.

    The parent engine must be constructed with ``transmission_mode='constant'``
    so its kernel does no in-kernel hazard work; this subclass supplies the
    infectivity from a dense PyTorch pass instead. ``prepass_mode`` selects what
    that dense pass computes.
    """

    prepass_mode = "constant"   # set on the class before construction

    def _static_step_forward(self) -> None:
        # Naive dense pre-pass over ALL N nodes, before the fused kernel reads
        # `infectivity` to build the pressure.
        i_mask = self.state == self._state_i
        if self.prepass_mode == "age_dependent":
            hz = lognormal_hazard_stable(
                torch.clamp(self.age, min=1e-10),
                self.model._mu_ir, self.model._sig_ir,
            )
            self.infectivity.copy_(
                torch.where(i_mask, self.model._beta_t * hz, 0.0)
            )
        else:
            self.infectivity.copy_(torch.where(i_mask, self.model._beta_t, 0.0))
        super()._static_step_forward()


def _model(mode):
    m = SEIRModel(beta=BETA, mean_ei=5.0, median_ei=4.0, mean_ir=7.5, median_ir=5.0)
    m.transmission_mode = mode
    return m


def bench(engine_factory, graph, trials=3):
    """Return NUPS for an engine factory (fresh engine per trial)."""
    N = graph.num_nodes
    tot_nups = tot_t = 0.0
    for tr in range(2 + trials):          # 2 warm-up trials
        eng = engine_factory(tr)
        eng.state[0] = eng._state_i
        eng.age[0] = 0.0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        steps = 0
        while eng.current_time < TF:
            eng.step()
            steps += eng.steps_per_launch
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if tr >= 2:
            tot_nups += N * steps
            tot_t += (t1 - t0)
        del eng
        torch.cuda.empty_cache()
    return tot_nups / tot_t if tot_t > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-nodes", type=int, default=1_000_000)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output", type=str,
                    default="results/agedep_prepass.csv")
    args = ap.parse_args()

    dev, N = args.device, args.num_nodes
    graphs = [
        ("Regular (d=8)", FixedDegreeGraph(N, 8, device=dev, seed=0)),
        ("BA (m=4)", BarabasiAlbertGraph(N, 4, device=dev, seed=0)),
    ]

    common = dict(device=dev, epsilon=EPSILON, tau_max=TAU_MAX,
                  steps_per_launch=STEPS_PER_LAUNCH, csr_strategy="thread")
    rows = []

    for gname, g in graphs:
        print(f"\n== {gname} (N={g.num_nodes:,}, E={g.num_edges:,}) ==", flush=True)

        # --- Pre-pass: kernel built in CONSTANT mode; dense pass supplies infectivity.
        def make_prepass(mode):
            def factory(tr):
                NaivePrepassFusedCG.prepass_mode = mode
                return NaivePrepassFusedCG(
                    g, _model("constant"), seed=12345 + tr * 100, **common)
            return factory

        pp_c = bench(make_prepass("constant"), g, args.trials)
        pp_a = bench(make_prepass("age_dependent"), g, args.trials)
        pp_slow = (pp_c / pp_a - 1.0) if pp_a > 0 else float("nan")
        print(f"  pre-pass : const={pp_c:,.0f}  age_dep={pp_a:,.0f}  "
              f"slowdown={pp_slow*100:+.1f}%")

        # --- In-kernel: production fused engine, mode baked at construction.
        def make_inkernel(mode):
            def factory(tr):
                return RenewalEngineFusedCUDAGraph(
                    g, _model(mode), seed=12345 + tr * 100, **common)
            return factory

        ik_c = bench(make_inkernel("constant"), g, args.trials)
        ik_a = bench(make_inkernel("age_dependent"), g, args.trials)
        ik_slow = (ik_c / ik_a - 1.0) if ik_a > 0 else float("nan")
        print(f"  in-kernel: const={ik_c:,.0f}  age_dep={ik_a:,.0f}  "
              f"slowdown={ik_slow*100:+.1f}%")

        rows.append((gname, "Pre-pass", pp_c, pp_a, pp_slow))
        rows.append((gname, "In-kernel", ik_c, ik_a, ik_slow))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("graph,implementation,constant_nups,age_dependent_nups,slowdown\n")
        for gname, impl, c, a, s in rows:
            f.write(f"{gname},{impl},{c:.1f},{a:.1f},{s:.4f}\n")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
