# Verification record · 27 October 2026

- Installed the course's isolated Pixi environment in `~/developer/ml/rl_gym/course_12_session`; parent project files preserved. Included its resolved lockfile. Apple Silicon / macOS 14+ target. Pixi emits a deprecation notice for the supported system-requirements table; installation succeeds.
- Passed 12 numerical tests: terminal targets, truncation bootstrap, masked selection, empty masks, DPO margin gradient, reference isolation, PPO clipping, constant GRPO groups, and n-step endpoints/intermediate targets.
- Completed two-iteration/episode smoke runs for all 12 focal methods plus optional actor-critic. Real raw metrics and checkpoints are bundled under results/smoke.
- Loaded and continued Q-learning, MC, DQN, PPO, SFT, DPO and GRPO checkpoints for one additional iteration/episode. These are warm restarts, not exact continuation.
- Executed frozen Taxi table evaluation and Blackjack Q-learning / FrozenLake SARSA transfer checks.
- Verified all HTML local links and checkpoint paths. Opened the index and DPO module in the app browser, switched to Quiz and confirmed answer reveal.
- The RGB-array display helper is tested separately; native GUI behavior and Windows are not validated.
- No full-budget convergence, multi-seed superiority, large-language-model alignment, Atari or MuJoCo benchmark claim is made. The included runs establish execution and artifact integrity only.

## Build checkpoints

1. Recovered local CartPole/Blackjack/Taxi sources and inspected Pixi configuration.
2. Created 12 modules, references, scaffolds and learning gates.
3. Added n-step bridge from user feedback; added rendering workaround.
4. Completed invariant, smoke and checkpoint-loading checks.
5. Packaged with source manifest and SHA-256 checksums; ZIP tested and extracted into a fresh validation directory.

## Module-command update

Updated every workshop command to dotted `python -m reference.…` syntax, added an explicit course-root notice to each HTML page and a reference package initializer, and updated harness/starter subprocess calls. All 12 numerical tests passed again; CartPole module execution and harness child-module invocation passed in the installed course environment. Parent project files remain unchanged.

## Module-command update

Updated every workshop command to dotted `python -m reference.…` syntax, added an explicit course-root notice to each HTML page and a reference package initializer, and updated harness/starter subprocess calls. All 12 numerical tests passed again; CartPole module execution and harness child-module invocation passed in the installed course environment. Parent project files remain unchanged.

## Introductory pacing correction

Split the first workshops into a standalone CartPole collection loop and tabular-only references. Included unchanged snapshots of the user’s working originals; removed later neural code from early module embeds. Added an explicit motivation for approximation in session 5.

Validated the new loop and MC/Q/SARSA entry points with two episodes each; verified early embedded references contain no PyTorch import or neural layers, and original source snapshots are byte-identical.

## Module-command update

Updated every workshop command to dotted `python -m reference.…` syntax, added an explicit course-root notice to each HTML page and a reference package initializer, and updated harness/starter subprocess calls. All 12 numerical tests passed again; CartPole module execution and harness child-module invocation passed in the installed course environment. Parent project files remain unchanged.

## Module-command update

Updated every workshop command to dotted `python -m reference.…` syntax, added an explicit course-root notice to each HTML page and a reference package initializer, and updated harness/starter subprocess calls. All 12 numerical tests passed again; CartPole module execution and harness child-module invocation passed in the installed course environment. Parent project files remain unchanged.
