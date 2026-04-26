# LucidWM

**Clean single-file implementations of world model algorithms.**

> When someone asks "how does Dreamer v3 actually work?", the answer should be:
> read `dreamer_v3_dmc.py`, it's 550 lines.

LucidWM applies the [CleanRL](https://github.com/vwxyzjn/cleanrl) philosophy to world models and model-based RL. Each algorithm is a single Python file with the complete model definition, training loop, evaluation, and logging.

## Three-Layer Architecture

```
lucidwm_components/     Layer 1: Reusable building blocks
├── rssm.py               GaussianRSSM, CategoricalRSSM
├── planners.py            CEMPlanner, MPPIPlanner, MCTSPlanner
├── networks.py            ConvEncoder, ConvDecoder, MLP, ResidualBlock
└── distributions.py       symlog, TwoHotDist, PercentileNormalizer, lambda_returns

lucidwm/                Layer 2: Single-file algorithms
├── wm_carracing.py        World Models (Ha & Schmidhuber, 2018)
├── planet_dmc.py          PlaNet (Hafner et al., 2019)
├── dreamer_v1_dmc.py      Dreamer v1 (Hafner et al., 2020)
├── dreamer_v2_atari.py    Dreamer v2 (Hafner et al., 2021)
├── dreamer_v3_dmc.py      Dreamer v3 (Hafner et al., 2023)
├── muzero_atari.py        MuZero (Schrittwieser et al., 2020)
├── tdmpc2_dmc.py          TD-MPC2 (Hansen et al., 2024)
└── diamond_atari.py       DIAMOND (Alonso et al., 2024)

lucidwm_utils/          Layer 3: Shared infrastructure
├── envs.py               Environment wrappers (DMC, Atari, Crafter, MetaWorld)
├── buffers.py             ReplayBuffer, PrioritizedReplayBuffer
├── metrics.py             Evaluation protocol
├── logger.py              W&B + TensorBoard logging
└── video.py               Video recording
```

**Layer 1** components are concrete implementations (not abstract base classes) extracted only when used by 3+ algorithms with <10% variation. They have no cross-dependencies.

**Layer 2** files import from Layers 1 and 3, but never from each other. Each file contains all algorithm-specific logic.

**Layer 3** is thin infrastructure (~300 LOC) that could be inlined if you wanted a truly standalone file.

## Install

```bash
pip install lucidwm                    # Core (DMControl)
pip install "lucidwm[atari]"           # + Atari
pip install "lucidwm[all]"             # Everything
```

## Usage

```bash
python -m lucidwm.dreamer_v3_dmc --env-id walker-walk --seed 1 --track
python -m lucidwm.tdmpc2_dmc --env-id cheetah-run --seed 1
python -m lucidwm.diamond_atari --env-id BreakoutNoFrameskip-v4 --seed 1
```

All implementations share the same CLI: `--env-id`, `--seed`, `--total-steps`, `--track`.

## Design Principles

1. **Single-file, self-contained.** Read one file, understand the whole algorithm.
2. **Readable over clever.** Paper variable names, shape comments on every tensor op.
3. **Conventions over configuration.** Same CLI, same wrappers, same logging. No YAML.
4. **Benchmarked.** Every implementation ships with reference curves and reproduction commands.

## Contributing

See the [design document](docs/design.md) for the full specification. To add a new algorithm:

1. Create a single `.py` file in `lucidwm/` following the template.
2. Use Layer 1 components where applicable.
3. Benchmark on ≥3 environments with ≥3 seeds.
4. Write a companion doc in `docs/`.
5. Quality bar: results within 1 std of published, all shapes documented, `ruff` clean.

## License

MIT
