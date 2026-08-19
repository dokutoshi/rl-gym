# rl-gym

Working code from the Reinforcement Learning track of the
[Data Science & Machine Learning Collaborative Learning Group](https://www.meetup.com/data-science-machine-learing-collaborative-learning-group/),
where I am one of the facilitators, with emphasis on Reinforcement Learning and Reasoning Models.

Built on the [Farama Foundation](https://farama.org/) stack — **Gymnasium** (the maintained
successor to OpenAI Gym) and **PettingZoo** for multi-agent environments.

## Curriculum

The sessions work through four sources in parallel:

- **[Gymnasium / PettingZoo](https://farama.org/)** — the environment API everything else targets
- **[Sutton & Barto, *Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book.html)** — the standard reference
- **[HuggingFace Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)**
- Notes from Stanford's Reinforcement Learning course

## Contents

| Path | Environment | Topic |
|---|---|---|
| `blackjack/` | `Blackjack-v1` | Tabular agent, trainer and inference kept in separate modules; Monte Carlo and TD compared (`Figure_1.png`, `Figure_3 TD.png`) |
| `cartpole.py` | `CartPole-v1` | Control baseline |
| `gridWorld/` | custom | A small hand-written environment plus a Gymnasium-compatible wrapper |
| `taxi/` | `Taxi-v3` | Action masking — restricting the action space to legal moves, with a before/after comparison (`_static/img/tutorials/taxi_v3_action_masking_comparison.png`) |
| `rl.ipynb` | — | Scratch notebook |

For anyone new to the API, `gridWorld/gridworld_wrapper.py` is a reasonable starting point — it
shows the minimum surface (`reset`, `step`, `observation_space`, `action_space`) needed for a
custom environment to work with the rest of the ecosystem.

## Running it

The environment is managed with [pixi](https://pixi.sh), so the lockfile pins everything:

```bash
pixi install
pixi run python cartpole.py
pixi run python blackjack/blackjack_trainer.py
pixi run jupyter lab            # rl.ipynb, taxi/action_masking_taxi.ipynb
```

`pixi.lock` is committed; `.pixi/` (the resolved environment, ~740 MB) is not — `pixi install`
rebuilds it.

## Notes

Some scripts open a render window. On a headless machine, set the environment's render mode to
`rgb_array` rather than `human`.

This is study and session material, not a library — it is organised for reading and discussion
rather than reuse.
