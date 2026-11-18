# Learning and training checkpoints

## Learning gates

Every HTML workshop has three persistent checkboxes and exportable notes. Gate A after session 4: explain MC versus TD, on/off-policy, terminal versus truncated, and show masked/unmasked Taxi results. Gate B after session 8: show a detached target, saved neural weights, and an explanation of replay versus old-policy likelihood ratios. Gate C after session 11: derive the DPO margin, inspect a zero-variance GRPO group, distinguish SFT / offline pairs / online rollouts. Gate D after session 12: deliver a reproducible comparison with limitations.

For every session save command, configuration, raw metrics, checkpoint, test outcome and a one-paragraph explanation. Facilitator scores each 0 (missing), 1 (runs but explanation incomplete), 2 (correct and explained); pass each category at 2 before increasing training budget. HTML progress is browser-local; exported notes and result files are the portable record.

## Saved artifacts and continuation

Tabular runs save `table.json` including Q and MC visit counts at completion. Neural runs save `checkpoint.pt` every `--save-every` episodes and at completion, with policy, critic, optimizer, episode and configuration. Language runs save logits, optimizer, fixed reference log probabilities and configuration at completion. All runs save `metrics.json` with raw scores, package versions, configuration and elapsed time. Avoid reusing output directories unless deliberately overwriting a run.

```sh
pixi run python -m reference.tabular --algo q --episodes 1000 --resume results/taxi/table.json --seed 100000 --out results/taxi_continued
pixi run python -m reference.train --algo dqn --episodes 100 --resume results/benchmark/dqn/0/checkpoint.pt --seed 100000 --out results/dqn_continued
pixi run python -m reference.tabular --algo evaluate --env Taxi-v4 --episodes 1000 --resume results/taxi/table.json --seed 200000 --out results/taxi_eval
```

Continuation is a **warm restart**, not bit-exact resumption: neural replay, target lag, RNG states, environment state, prior metrics and total step counters are not restored. The target is synchronized from the loaded policy. Use matching algorithm, environment, feature shape and hyperparameters. Tabular counts are restored; new seeds avoid repeating the same hands. DPO/GRPO preserve the original fixed reference on restart. Do not load checkpoints from untrusted sources.

`evaluate` freezes tabular action values; without `--resume` it evaluates a tied-zero policy, not a trained agent. Neural held-out evaluation is a capstone implementation exercise, not supplied as a misleading training-return benchmark. Unit tests and smoke runs do not establish convergence.


## n-step bridge after session 4

Sutton & Barto second edition Chapter 7 unifies MC and one-step TD. Run `pixi run python -m reference.nstep` and the endpoint tests. Optional session 4A integrates n-step SARSA; n=1 is one-step SARSA, a terminal reward tail is MC. Actions still change every step. Keep eligibility traces / TD(λ), Chapter 12, as a later extension.


## Introductory pacing correction

Sessions 1–4 use isolated loop/tabular references with no neural-network code. Your original CartPole, Blackjack and Taxi sources are included as unchanged reading snapshots under `examples_original/`; run the Blackjack originals from the parent project, where their plotting dependencies are installed. Session 1 collects experience without learning. Sessions 2–4 learn lookup-table values. Session 5 motivates shared parameters for continuous observations and introduces a linear approximation; session 6 introduces a neural network as DQN’s value approximator. Neural networks are a representation choice, not a prerequisite for reinforcement learning.
