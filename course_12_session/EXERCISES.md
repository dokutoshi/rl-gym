# Exercises and acceptance criteria

For each exercise copy the reference into a personal branch, implement the change, add one failure-catching assertion, and save metrics. A runnable starter launches a baseline; its placeholder target is a warm-up, not the completed algorithm.

## 01. Random-policy rollout • CartPole

**Implement:** Add termination/truncation counts and a pole-angle heuristic policy.

**Accept:** Total episodes equals both end counters plus any explicit collector cutoffs; identical seeds reproduce the random run.

**Between meetings:** Compare 30 random episodes across seeds 0–4. Record lengths and why each episode ended. Change max-steps from 500 to 50; explain why this changes the measured return.

**Submit:** code diff, passing assertion, raw metrics and checkpoint, 150-word interpretation.

## 02. First-visit Monte Carlo control • Blackjack

**Implement:** Extract first-visit returns into a function; hand-test a repeated state/action trajectory.

**Accept:** For rewards 1 then 2 and gamma=1, the first-visit target is 3, not 2; counts increase once per episode.

**Between meetings:** Train 50,000 hands between sessions; compare epsilon .05/.1/.2 over five seeds. Evaluate the frozen greedy table on separate seeds. Report mean return and win/draw/loss frequencies.

**Submit:** code diff, passing assertion, raw metrics and checkpoint, 150-word interpretation.

## 03. Masked Q-learning • Taxi-v4

**Implement:** Patch a copy of the original Taxi example so terminal transitions update; return Q table; count masked invalid actions.

**Accept:** The terminal +20 transition moves a zero Q value to 2 with alpha=.1; masked action selection never chooses a zero-mask action.

**Between meetings:** Paired seeds 0–4, 5,000 episodes, masked versus unmasked, same budgets. Report both exploration returns and separate greedy evaluation. Count invalid actions; do not claim every mask guarantees safety outside this task.

**Submit:** code diff, passing assertion, raw metrics and checkpoint, 150-word interpretation.

## 04. SARSA • CliffWalking and FrozenLake

**Implement:** Run Q-learning and SARSA with identical Cliff seeds; implement Expected SARSA as optional work.

**Accept:** SARSA target uses the sampled next action, Q-learning uses a maximum; show a hand example where targets differ.

**Between meetings:** Cliff: compare training return versus greedy route. FrozenLake: run at least 10,000 episodes and evaluate success probability, not just a single lucky trajectory. Keep map and slipperiness fixed.

**Submit:** code diff, passing assertion, raw metrics and checkpoint, 150-word interpretation.

## 05. Semi-gradient Q-learning • linear approximation

**Implement:** Add running observation normalization and freeze its statistics at evaluation.

**Accept:** A zero TD error leaves weights unchanged; terminal next-value does not affect the target.

**Between meetings:** Run lr .0003/.001/.01 over five seeds. Record TD loss and episode return. Add normalized features as a homework ablation. A short run is a plumbing check, not convergence evidence.

**Submit:** code diff, passing assertion, raw metrics and checkpoint, 150-word interpretation.

## 06. DQN • replay and target networks

**Implement:** Log TD loss, replay size and target sync; implement Double DQN selection/evaluation separation.

**Accept:** Target weights have no gradients; sync occurs at the declared optimizer-update interval.

**Between meetings:** Between sessions train 300–1,000 episodes, five seeds. Ablate replay or target lag one at a time. Report environment steps, return, wall time and evaluation policy.

**Submit:** code diff, passing assertion, raw metrics and checkpoint, 150-word interpretation.

## 07. REINFORCE • policy gradients

**Implement:** Add a scalar return baseline; compare gradient variance on a fixed batch before and after.

**Accept:** Zero advantage produces zero policy-loss gradient; baseline is action independent.

**Between meetings:** Compare raw returns with a running scalar baseline as an exercise. Run 300 episodes and five seeds; plot gradient norms and return variability.

**Submit:** code diff, passing assertion, raw metrics and checkpoint, 150-word interpretation.

## 08. PPO-Clip • actor-critic bridge

**Implement:** Log clip fraction and approximate KL; optional 8A: run --algo ac and compare actor/critic losses.

**Accept:** Old log probabilities and advantages stay fixed across epochs; actor and critic both receive gradients.

**Between meetings:** In class run two short episodes and inspect ratios. Between sessions compare 1 versus 4 update epochs and seeds 0–4. Add approximate KL logging, advantage normalization and GAE in an optional session.

**Submit:** code diff, passing assertion, raw metrics and checkpoint, 150-word interpretation.

## 09. Supervised fine-tuning • pretraining → RLHF

**Implement:** Add noisy demonstrations, a dataset manifest and explicit train/validation split. Optional 9A: sequence cross entropy with prompt padding masks.

**Accept:** Only target completion tokens contribute to the sequence extension; report why unseen lookup IDs cannot generalize.

**Between meetings:** Compare demonstration noise levels in an exercise and record held-out prompt accuracy after designing a feature-sharing model. The supplied lookup model cannot generalize to unseen prompt IDs.

**Submit:** code diff, passing assertion, raw metrics and checkpoint, 150-word interpretation.

## 10. Direct Preference Optimization

**Implement:** Extract DPO loss; test equal margins, swapped labels and frozen reference gradients.

**Accept:** Equal relative margin produces log(2); swapping labels reverses the gradient direction.

**Between meetings:** Sweep beta .05/.1/.5, flip 20% of preference labels, and compare preference accuracy and reference KL. Toy correctness is objective verification, not evidence of human preference alignment.

**Submit:** code diff, passing assertion, raw metrics and checkpoint, 150-word interpretation.

## 11. Group Relative Policy Optimization

**Implement:** Expose group size as an argument, log zero-variance group fraction and exact categorical KL.

**Accept:** Constant-reward group advantages are finite zeros; old policy differs from fixed reference after updates.

**Between meetings:** Vary group size 2/8/16 as an exercise; measure zero-variance groups, reward and exact categorical KL. Compare rollout cost. Uniformly correct or incorrect groups have zero relative advantage.

**Submit:** code diff, passing assertion, raw metrics and checkpoint, 150-word interpretation.

## 12. Monte Carlo policy evaluation • capstone harness

**Implement:** Implement a frozen neural-policy evaluator with separate seeds; aggregate paired seed differences for the capstone.

**Accept:** Evaluation changes no weights. Report training-seed SD separately from within-policy episode variation.

**Between meetings:** Capstone: Taxi masked/unmasked or Cliff SARSA/Q comparison; or DPO/GRPO toy objective audit. Submit raw metrics, protocol, confidence intervals, architecture and failure analysis. Do not compare toy rewards to published LLM benchmarks.

**Submit:** code diff, passing assertion, raw metrics and checkpoint, 150-word interpretation.


## n-step bridge after session 4

Sutton & Barto second edition Chapter 7 unifies MC and one-step TD. Run `pixi run python -m reference.nstep` and the endpoint tests. Optional session 4A integrates n-step SARSA; n=1 is one-step SARSA, a terminal reward tail is MC. Actions still change every step. Keep eligibility traces / TD(λ), Chapter 12, as a later extension.


## Introductory pacing correction

Sessions 1–4 use isolated loop/tabular references with no neural-network code. Your original CartPole, Blackjack and Taxi sources are included as unchanged reading snapshots under `examples_original/`; run the Blackjack originals from the parent project, where their plotting dependencies are installed. Session 1 collects experience without learning. Sessions 2–4 learn lookup-table values. Session 5 motivates shared parameters for continuous observations and introduces a linear approximation; session 6 introduces a neural network as DQN’s value approximator. Neural networks are a representation choice, not a prerequisite for reinforcement learning.
