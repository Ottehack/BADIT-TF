# Experiment coverage

The catalog exposes the experiment families below through `badit-tf list`.
Preparation and aggregation scripts are retained separately so an Agent can
inspect every transformation instead of relying on an opaque shell queue.

| Family | Purpose | Main runner(s) | Main aggregation |
|---|---|---|---|
| P0 | injection/initialization smoke | `run_p0.py` | direct JSON evidence |
| P1/H0 | Task-Fisher objective, damping, fidelity | `run_p1_collect.py`, `run_p1_fidelity.py`, `run_h0_fidelity.py` | `aggregate_p1*.py`, `aggregate_h0.py` |
| P2 | multi-task TF/GG | `run_p2_train.py` | `aggregate_p2.py` |
| P3 | continual TF/GG | `run_p3_sequential.py` | `aggregate_p3*.py` |
| H1/H2 | one-factor/joint selection | P2/P3 runners | H1/H2 preparation and selection scripts |
| H3/H4 | setting/backbone transfer | P2/P3 runners | `aggregate_h3_confirmation.py`, `aggregate_h4_transfer.py` |
| M0/M1 | primary multi-task/continual claims | P2/P3 runners | `aggregate_m0_m1_official.py` |
| M2 | fidelity and assignment comparators | `run_m2_collect.py`, `run_m2_fidelity.py` | `aggregate_m2*.py` |
| M3 | local-to-failure intervention scale | `run_m3_deployment_scale.py` | `aggregate_m3*.py` |
| M4 | objective decomposition | `run_m4_collect.py`, `run_m4_decomposition.py` | `aggregate_m4*.py` |
| M5/Table VI | contiguous/no-capacity ablations | M2 and no-capacity runners | `aggregate_m5_contiguous.py`, TPAMI aggregate |
| M6 | discovery transfer | P2/P3 runners with generated configs | `prepare_m6_discovery_transfer.py` |
| M7 | partition stability | `run_m7_dog_collect.py`, `run_m7_partition_stability.py` | `aggregate_m7*.py` |
| M8/Table IX | ability/intervention | `run_m8_ability.py` | `aggregate_m8*.py` |
| M9/Table XI | calibration and throughput cost | `run_m9_throughput.py` | `aggregate_m9*.py`, `aggregate_table_xi_cost_replays.py` |
| Table I | single-task controls | generated P2/P3-compatible runs | `aggregate_table_i_single_task_controls.py` |

Some preparation scripts generate a grid of resolved configurations rather than
performing training themselves. Generated files belong under
`experiments/configs/` or `.runs/` and must be reviewed before launch.

The legacy internal PM/OSS queueing, host recovery, evacuation, and migration
scripts are deliberately excluded. They are operational history, not part of
the scientific method, and contained non-portable infrastructure assumptions.

