# Reading and source map

Sutton, R. S. & Barto, A. G. (2018), *Reinforcement Learning: An Introduction*, second edition, MIT Press. [Author book page](http://incompleteideas.net/book/the-book-2nd.html). Section citations in modules refer to this edition, not the older draft or first edition. The author page could not be fetched during validation; the citation is provided for learner access. No textbook content is reproduced.

- Sessions 1–4: ch. 3 (MDPs), ch. 5 (Monte Carlo), §§6.4–6.6 (SARSA, Q-learning, Expected SARSA).
- Sessions 5–6: chs. 9–11 (approximation and off-policy stability), §16.5 (deep RL).
- Sessions 7–8: ch. 13 (policy gradients and actor-critic). PPO itself is in the paper below.
- Sessions 9–11: ch. 13 provides a conceptual connection only. The 2018 book does not contain DPO or GRPO. Read the primary papers.
- Session 12: §5.1 Monte Carlo prediction and §3.3 returns.

[Gymnasium Taxi-v4](https://gymnasium.farama.org/v1.3.0/environments/toy_text/taxi/) and [action masking tutorial](https://gymnasium.farama.org/tutorials/training_agents/action_masking_taxi/). The live tutorial reports a newer compatible version; the course pins and tests 1.3.0. The user's local taxi example is preserved in its original location. Course-specific corrections are explained in README.

[Mnih et al. (2015), Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236): replay and target-network motivation, not a claim that our CartPole implementation reproduces Atari results.

[Schulman et al. (2017), Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), §3: clipped surrogate. [Ouyang et al. (2022), Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), §3: demonstrations, reward modeling and RLHF pipeline.

[Rafailov et al. (2023), Direct Preference Optimization](https://arxiv.org/abs/2305.18290), §§3–4: KL-regularized preference optimization and objective. [Shao et al. (2024), DeepSeekMath](https://arxiv.org/abs/2402.03300), §4: group-relative policy optimization. Read paper definitions before interpreting the toy experiments as sequence-level implementations.

Environment progression: [toy text](https://gymnasium.farama.org/environments/toy_text/), [Box2D](https://gymnasium.farama.org/environments/box2d/), [MuJoCo](https://gymnasium.farama.org/environments/mujoco/), [ALE](https://ale.farama.org/environments/). Advanced environments are optional; environment complexity and algorithm complexity are separate curriculum choices.


## n-step bridge after session 4

Sutton & Barto second edition Chapter 7 unifies MC and one-step TD. Run `pixi run python -m reference.nstep` and the endpoint tests. Optional session 4A integrates n-step SARSA; n=1 is one-step SARSA, a terminal reward tail is MC. Actions still change every step. Keep eligibility traces / TD(λ), Chapter 12, as a later extension.


## Where neural networks enter

The second edition already includes §9.7, Nonlinear Function Approximation: Artificial Neural Networks; §16.5 discusses Human-level Video Game Play and §16.6 covers Go. Use §9.4 for the linear model before §9.7 for neural approximation. See the [second-edition text hosted by CMU](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf). These foundations are in the 2018 edition; later sources are needed for DPO/GRPO.

Stanford [CS224R: Deep Reinforcement Learning](https://cs224r.stanford.edu/) emphasizes practical neural RL and assumes familiarity with deep learning. [CS234: Reinforcement Learning](https://bulletin.stanford.edu/courses/2190451) is the broader RL course and also includes deep RL. Our first workshops build the loop/tabular prerequisites before following the deep-RL progression.
