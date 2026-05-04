# LucidWM

**Single-file educational implementations of world model algorithms.**

Every world model paper ships its own codebase with its own conventions. Dreamer is JAX with OmegaConf. TD-MPC2 is PyTorch with a different layout. DINO-WM uses Hydra. LeWM is something else entirely. If you want to understand how any of these algorithms actually work, you're fighting config systems and module graphs before you get to the algorithm.

LucidWM puts each algorithm in one Python file. You read one file, you understand the whole thing. All implementations are PyTorch and share a similar style, but some are more faithful than others and several include practical simplifications compared to the original papers and repos.

## Implemented Algorithms

| Algorithm | File | Family | What It Does |
|-----------|------|--------|-------------|
| World Models | `wm_carracing.py` | VAE+RNN | VAE + MDN-RNN + CMA-ES controller, with a simplified training pipeline |
| Dreamer V1 | `dreamer_dmcontrol.py` | Latent imagination | Gaussian RSSM + imagination actor-critic on DMControl pixels |
| DIAMOND | `diamond_atari.py` | Diffusion world model | Diffusion next-frame model + reward/termination CNN-LSTM + imagined actor-critic on Atari |
| TD-MPC2 | `tdmpc2_dmcontrol.py` | Implicit | No decoder. SimNorm latent + MPPI planning. Online + offline (HuggingFace) |
| PWM | `pwm_dmcontrol.py` | Implicit | Same WM as TD-MPC2, but learns policy via first-order gradients through frozen dynamics |
| PWM (MuJoCo) | `pwm_mujoco.py` | Implicit | PWM trained on Minari offline datasets (HalfCheetah, Hopper, Ant, etc.) |
| DINO-WM | `dinowm_pusht.py` | JEPA | Frozen DINOv2 backbone + causal ViT dynamics. Zero-shot goal planning |
| LeWM | `lewm_pusht.py` | JEPA | Learned encoder + history-conditioned predictor + SIGReg. Faithful where practical |

## Architecture

```text
lucidwm_components/     Layer 1: Reusable building blocks
├── networks.py            ConvEncoder, ConvDecoder, MLP, ResidualBlock
└── distribution.py        symlog, symexp, SimNorm, TwoHotDist

lucidwm/                Layer 2: Single-file algorithms
├── wm_carracing.py        651 LOC, fully implemented
├── dreamer_dmcontrol.py   Dreamer V1-style implementation for DMControl pixels
├── diamond_atari.py       DIAMOND-style implementation for Atari RGB observations
├── tdmpc2_dmcontrol.py    TD-MPC2-style implementation (online + offline)
├── pwm_dmcontrol.py       PWM-style implementation
├── pwm_mujoco.py          PWM on Minari datasets
├── dinowm_pusht.py        DINO-WM-style implementation
├── lewm_pusht.py          LeWM-style implementation

lucidwm_utils/          Layer 3: Shared infrastructure
├── envs.py               make_env() with state/pixel support for DMC, Atari, MuJoCo
├── buffers.py             ReplayBuffer
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

# Dreamer V1 on DMControl pixels
python -m lucidwm.dreamer_dmcontrol --env-id walker-walk --seed 1 --track

# DIAMOND on Atari
python -m lucidwm.diamond_atari --env-id BreakoutNoFrameskip-v4 --seed 1 --track

# TD-MPC2: online training on DMControl
python -m lucidwm.tdmpc2_dmcontrol --env-id walker-walk --seed 1 --track

# TD-MPC2: offline training on released datasets
python -m lucidwm.tdmpc2_dmcontrol --offline --dataset mt30

# PWM: learn policy via FoG through frozen world model
python -m lucidwm.pwm_dmcontrol --dataset mt30 --env-id walker-walk

# PWM on MuJoCo via Minari
python -m lucidwm.pwm_mujoco --dataset mujoco/halfcheetah/medium-v0 --eval

# DINO-WM: train dynamics on frozen DINOv2 features
python -m lucidwm.dinowm_pusht --data-dir ./data/pusht --action-dim 2

# LeWM: end-to-end JEPA from pixels
python -m lucidwm.lewm_pusht --data-path ./data/pusht_expert_train.h5 --action-dim 2
```

## Fidelity

These are single-file educational implementations, not exact ports of the original repos.
Some algorithms are close to the published method, while others intentionally simplify data loading,
training structure, or planning to keep the code readable in one file.

## What's Covered

The library spans five distinct paradigm families:

**VAE + RNN** (World Models): The original. Learn a compressed visual model and a recurrent dynamics model separately, then optimize a tiny controller entirely inside the dream.

**Latent Imagination / RSSM** (Dreamer): Learn a recurrent stochastic state-space model with reconstruction and reward prediction, then optimize an actor and value function directly through imagined latent rollouts.

**Diffusion World Model** (DIAMOND): Predict future Atari frames directly in pixel space with an EDM-style denoiser, then train a recurrent policy and value function entirely inside imagined rollouts.

**Implicit / SimNorm** (TD-MPC2, PWM): No decoder, no reconstruction. Predict next latent, reward, and value directly. SimNorm prevents latent blowup. TD-MPC2 plans with MPPI at test time. PWM shows you can skip planning entirely and learn a better policy by backpropagating through the frozen dynamics.

**Frozen Foundation Model** (DINO-WM): Don't learn perception at all. Freeze DINOv2, learn only dynamics on top of its patch features. Plan by optimizing action sequences to reach goal embeddings.

**Learned JEPA** (LeWM): Learn perception end-to-end but prevent collapse with SIGReg (a statistical test enforcing Gaussian latents). Two loss terms total. ~15M params on a single GPU in hours.

## Contributing

1. Create a single `.py` file in `lucidwm/`
2. Follow the file anatomy: docstring, config, models, training, main loop
3. Use Layer 1 components where applicable
4. Benchmark on ≥3 environments with ≥3 seeds
5. Write a companion doc in `docs/`

Quality bar: results within 1 std of published, all tensor shapes documented, `ruff` clean.

## License

MIT
