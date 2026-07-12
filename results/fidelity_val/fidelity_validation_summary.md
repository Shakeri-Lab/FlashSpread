# FlashSpread fidelity re-validation

**Overall: PASS**  (11 pass, 0 fail, 0 skip)

Checks the post Phase-0/1 simulator against exact Gillespie, vs the JOCS structural-bias floor (~6% peak-I, ~7% final-R, epsilon-independent).

Epsilon-sweep reference: `matlab_exact_gillespie`

| Section | Check | Value | Band | Result |
|---|---|---|---|---|
| eps-sweep | err_peak_I @ eps=0.03 | 0.0637 | <= 0.1 | PASS |
| eps-sweep | err_final_R @ eps=0.03 | 0.0685 | <= 0.12 | PASS |
| eps-sweep | err_peak_I spread across eps (independence) | 0.0115 | <= 0.03 | PASS |
| multi-graph | peak-I err er N=1000 | 0.0021 | <= 0.1 | PASS |
| multi-graph | peak-I err er N=10000 | 0.0024 | <= 0.1 | PASS |
| multi-graph | peak-I err ba N=1000 | 0.0084 | <= 0.1 | PASS |
| multi-graph | peak-I err ba N=10000 | 0.0014 | <= 0.1 | PASS |
| multi-graph | peak-I err fixed N=1000 | 0.0019 | <= 0.1 | PASS |
| multi-graph | peak-I err fixed N=10000 | 0.001 | <= 0.1 | PASS |
| markovian | SIS mean-traj L2 | 0.0141 | <= 0.05 | PASS |
| markovian | SIR mean-traj L2 | 0.012 | <= 0.05 | PASS |
