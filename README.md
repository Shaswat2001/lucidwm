# LucidWM

**Single-file implementations of world model algorithms.**

Every world model paper ships its own codebase with its own conventions. Dreamer is JAX with OmegaConf. TD-MPC2 is PyTorch with a different layout. DINO-WM uses Hydra. LeWM is something else entirely. If you want to understand how any of these algorithms actually work, you're fighting config systems and module graphs before you get to the algorithm.

LucidWM puts each algorithm in one Python file. You read one file, you understand the whole thing. All implementations are PyTorch, use the same CLI, the same logging, the same evaluation protocol.

## Implemented Algorithms

| Algorithm | File | Family | What It Does |
|-----------|------|--------|-------------|
| World Models | `wm_carracing.py` | VAE+RNN | VAE + MDN-RNN + CMA-ES controller in dream |
| TD-MPC2 | `tdmpc2_dmc.py` | Implicit | No decoder. SimNorm latent + MPPI planning. Online + offline (HuggingFace) |
| PWM | `pwm_dmc.py` | Implicit | Same WM as TD-MPC2, but learns policy via first-order gradients through frozen dynamics |
| PWM (MuJoCo) | `pwm_mujoco.py` | Implicit | PWM trained on Minari offline datasets (HalfCheetah, Hopper, Ant, etc.) |
| DINO-WM | `dinowm_pusht.py` | JEPA | Frozen DINOv2 backbone + causal ViT dynamics. Zero-shot goal planning |
| LeWM | `lewm_pusht.py` | JEPA | Learned ViT-Tiny encoder + SIGReg. Two loss terms. ~15M params, single GPU |

## Architecture

```
lucidwm_components/     Layer 1: Reusable building blocks
├── networks.py            ConvEncoder, ConvDecoder, MLP, ResidualBlock
└── distributions.py       symlog, TwoHotDist, PercentileNormalizer, lambda_returns

lucidwm/                Layer 2: Single-file algorithms
├── wm_carracing.py        651 LOC, fully implemented
├── tdmpc2_dmc.py          970 LOC, fully implemented (online + offline)
├── pwm_dmc.py             734 LOC, fully implemented
├── pwm_mujoco.py          373 LOC, fully implemented (Minari datasets)
├── dinowm_pusht.py        634 LOC, fully implemented
├── lewm_pusht.py          580 LOC, fully implemented

lucidwm_utils/          Layer 3: Shared infrastructure
├── envs.py               make_env() with state/pixel support for DMC, Atari, MuJoCo
├── buffers.py             ReplayBuffer, PrioritizedReplayBuffer
├── metrics.py             Standardized evaluation protocol
├── logger.py              W&B + TensorBoard
└── video.py               Video recording
```

**Layer 1** components are concrete implementations (not abstract base classes) extracted when used by 3+ algorithms with <10% variation. They have no cross-dependencies.

**Layer 2** files import from Layers 1 and 3 but never from each other. Each contains all algorithm-specific logic.

## Install

```bash
pip install lucidwm                        # Core (DMControl)
pip install "lucidwm[atari]"               # + Atari
pip install "lucidwm[minari]"              # + Minari offline datasets
pip install "lucidwm[offline]"             # + HuggingFace datasets (TD-MPC2)
pip install "lucidwm[all]"                 # Everything
```

## Usage

```bash
# World Models: 3-phase training on CarRacing
python -m lucidwm.wm_carracing --seed 1

# TD-MPC2: online training on DMControl
python -m lucidwm.tdmpc2_dmc --env-id walker-walk --seed 1 --track

# TD-MPC2: offline training on released datasets
python -m lucidwm.tdmpc2_dmc --offline --dataset mt30

# PWM: learn policy via FoG through frozen world model
python -m lucidwm.pwm_dmc --dataset mt30 --env-id walker-walk

# PWM on MuJoCo via Minari
python -m lucidwm.pwm_mujoco --dataset mujoco/halfcheetah/medium-v0 --eval

# DINO-WM: train dynamics on frozen DINOv2 features
python -m lucidwm.dinowm_pusht --data-dir ./data/pusht --action-dim 2

# LeWM: end-to-end JEPA from pixels
python -m lucidwm.lewm_pusht --data-dir ./data/pusht --action-dim 2
```

## What's Covered

The library spans five distinct paradigm families:

**VAE + RNN** (World Models): The original. Learn a compressed visual model and a recurrent dynamics model separately, then optimize a tiny controller entirely inside the dream.

**Implicit / SimNorm** (TD-MPC2, PWM): No decoder, no reconstruction. Predict next latent, reward, and value directly. SimNorm prevents latent blowup. TD-MPC2 plans with MPPI at test time. PWM shows you can skip planning entirely and learn a better policy by backpropagating through the frozen dynamics.

**Frozen Foundation Model** (DINO-WM): Don't learn perception at all. Freeze DINOv2, learn only dynamics on top of its patch features. Plan by optimizing action sequences to reach goal embeddings.

**Learned JEPA** (LeWM): Learn perception end-to-end but prevent collapse with SIGReg (a statistical test enforcing Gaussian latents). Two loss terms total. ~15M params on a single GPU in hours.

**RSSM / Dreamer** (stubs): The Dreamer line with recurrent state-space models and actor-critic in imagination. Stubs are scaffolded with full docstrings and component wiring.

## Contributing

1. Create a single `.py` file in `lucidwm/`
2. Follow the file anatomy: docstring, config, models, training, main loop
3. Use Layer 1 components where applicable
4. Benchmark on ≥3 environments with ≥3 seeds
5. Write a companion doc in `docs/`

Quality bar: results within 1 std of published, all tensor shapes documented, `ruff` clean.

## License

MIT
