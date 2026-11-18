# RL → DPO hands-on course

12 × 90-minute workshops over 5–6 months. Open `index.html` for the course; all modules work offline after download. Python requires local dependencies. Prerequisites: Python, NumPy, probability, and gradients for sessions 5 onward. No GPU or API key required.

## Pixi setup (your existing project)

The course is intentionally isolated inside `~/developer/ml/rl_gym/course_12_session`; the parent Pixi manifest and lockfile are preserved.

```sh
cd ~/developer/ml/rl_gym/course_12_session
pixi install
pixi run test
pixi run smoke
pixi run serve
```

Open http://localhost:8000 in your browser. Or open `index.html` directly; if browser storage is restricted on local files, use the server and export notes. First installation needs internet. HTML contains no CDN or external script dependencies. Dependencies are exact top-level versions, and a tested Apple Silicon `pixi.lock` is included. The manifest targets macOS 14+ on Apple Silicon; other platforms need a separate manifest solve. Python 3.12 is the course baseline; the parent project currently uses 3.11.

## Working folder and module commands

All workshop commands assume the current working directory is `~/developer/ml/rl_gym/course_12_session` (the folder name is **course_12_session**).

```sh
cd ~/developer/ml/rl_gym/course_12_session
pixi run python -m reference.cartpole_loop --episodes 20 --out results/01_loop
```

`-m` takes a dotted module name, with no slash or `.py`. `reference/__init__.py` marks the reference directory as a package. Module execution lets sibling package imports resolve from the course root. `--out results/01_loop` is still relative to your current working directory; `-m` does not change it. Direct script execution can work for self-contained files, but this pack consistently teaches module execution.

For the original Blackjack examples, return to the **parent** project and its Pixi environment:

```sh
cd ~/developer/ml/rl_gym
pixi run python -m blackjack.blackjack_trainer
```

## First workshop / longer runs

```sh
pixi run python -m reference.cartpole_loop --episodes 20 --out results/cartpole
pixi run python -m reference.tabular --algo mc --episodes 50000 --out results/blackjack
pixi run python -m reference.tabular --algo q --episodes 5000 --out results/taxi
pixi run python -m reference.tabular --algo q --env Blackjack-v1 --episodes 50000 --out results/blackjack_q
pixi run python -m reference.tabular --algo sarsa --env FrozenLake-v1 --episodes 10000 --out results/lake
pixi run python -m reference.harness --algo dqn --episodes 300 --seeds 0 1 2 3 4
```

Core session commands appear in each HTML module. `--episodes` means optimizer iterations for SFT/DPO/GRPO and environment episodes for the other algorithms. `--lr .01` is a smoke default, not a tuned recommendation: try .0003–.001 for DQN/PPO and .1 for tabular control. Heavy training belongs between meetings. Results vary; no solved-environment claim accompanies these short runs.

## Files

- `modules/`: 12 standalone HTML workshops, each with five requested subject views plus Workshop and Quiz.
- `reference/train.py`: complete CPU teaching implementations; `harness.py`: seed sweeps and smoke runs.
- `starters/`: runnable scaffolds. Their baseline target intentionally lacks bootstrapping; implement the session exercise in a copied reference branch. The launcher runs the reference baseline, not the unfinished student function.
- `tests/`: numerical invariant tests; `EXERCISES.md`: objective-specific work and acceptance checks.
- `CHECKPOINTS.md`: learning gates, artifact locations and continuation limits.
- `results/smoke/`: actual short-run outputs from validation, not course benchmark baselines.
- `SOURCES.md`, `FACILITATOR.md`, `VALIDATION.md`: references, teaching guide and verification record.

## Honest scope

The linear learner is a simple semi-gradient approximation baseline. DQN uses replay, a target network and Huber loss. PPO is a small episodic clipped actor-critic without GAE/vectorization, not production PPO. Actor-critic has an optional dedicated workshop. SFT, DPO and GRPO use a four-prompt categorical one-token model with synthetic deterministic labels/rewards: no pretrained transformer, tokenizer, real human preference dataset or large-model benchmark is bundled. The complete RLHF pipeline is taught architecturally; training a separate reward model and doing sequence-level PPO is an optional extension. Offline code examples cover fixed preference training; replay makes DQN off-policy, not automatically an offline RL dataset method.

## Provenance

No `/mnt/data/rl_curriculum` or prior ZIP was present. The chat reference returned placeholder answers, so this is a rebuild from the requested scope plus the locally available examples. The original `taxi/action_masking_taxi.py` informed the three masking points. It skips terminal updates; the course corrects that and separates termination from truncation. Original local files remain untouched. External sources are linked rather than repackaged.

## Display windows on macOS / Windows

The parent README's existing workaround is `render_mode="rgb_array"` instead of `"human"`. Human mode opens a native Pygame/SDL render window; it does not require launching another terminal. RGB-array mode returns pixels for inline notebook display or saving. It still requires the classic-control rendering dependencies. Core training uses no render mode at all.

```sh
# In the course directory, using the same terminal:
pixi run python -m reference.render_cartpole
# Saves results/cartpole.png; open it in your browser/editor.
pixi run python -m reference.render_cartpole --mode human
```

For a notebook: make the environment with `render_mode="rgb_array"`, then use `display(PIL.Image.fromarray(env.render()))` if Pillow/IPython are installed in that notebook environment. Select the render mode when constructing the environment. See the [Gymnasium rendering API](https://gymnasium.farama.org/api/env/). This pack tests PNG output on macOS; Windows GUI behavior is not verified, and the included Pixi manifest targets Apple Silicon. The parent `pixi run python cartpole.py` command remains available from the parent folder.


## Introductory pacing correction

Sessions 1–4 use isolated loop/tabular references with no neural-network code. Your original CartPole, Blackjack and Taxi sources are included as unchanged reading snapshots under `examples_original/`; run the Blackjack originals from the parent project, where their plotting dependencies are installed. Session 1 collects experience without learning. Sessions 2–4 learn lookup-table values. Session 5 motivates shared parameters for continuous observations and introduces a linear approximation; session 6 introduces a neural network as DQN’s value approximator. Neural networks are a representation choice, not a prerequisite for reinforcement learning.
