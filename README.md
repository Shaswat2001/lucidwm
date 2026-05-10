# LucidWM

**Educational implementations of world model algorithms — one folder per algorithm, one concept per file.**

Every world model paper ships its own codebase with its own conventions. Dreamer is JAX with OmegaConf. TD-MPC2 is PyTorch with a different layout. DINO-WM uses Hydra. LeWM is something else entirely. If you want to understand how any of these algorithms actually work, you're fighting config systems and module graphs before you get to the algorithm.

LucidWM gives each algorithm its own small package — a folder with a handful of focused files. You read one folder, you understand the whole algorithm. All implementations are PyTorch and share a common style, but some are more faithful to the original papers than others and several include practical simplifications.

## Implemented Algorithms

| Algorithm | Folder | Family | What It Does |
|-----------|--------|--------|-------------|
| World Models | `wm/` | VAE+RNN | VAE + MDN-RNN + CMA-ES controller |
| Dreamer V1 | `dreamer/` | Latent imagination | Gaussian RSSM + imagination actor-critic on DMControl |
| Dreamer V2 | `dreamerv2/` | Latent imagination | Categorical RSSM + straight-through gradients on Atari |
| DIAMOND | `diamond/` | Diffusion world model | Diffusion next-frame model + reward/termination CNN-LSTM + imagined actor-critic on Atari |
| TD-MPC2 | `tdmpc2/` | Implicit | No decoder. SimNorm latent + MPPI planning. Online + offline (HuggingFace) |
| PWM | `pwm/` | Implicit | Same WM as TD-MPC2, but policy learned via first-order gradients through frozen dynamics. `--dataset dmcontrol` or `--dataset mujoco` |
| DINO-WM | `dinowm/` | JEPA | Frozen DINOv2 backbone + causal ViT dynamics. Zero-shot goal planning via CEM |
| LeWM | `leworldmodel/` | JEPA | Learned encoder + history-conditioned predictor + SIGReg |

## Architecture

```text
lucidwm_components/         Layer 1: Reusable building blocks (no cross-deps)
├── networks.py               MLP, ConvEncoder, ConvDecoder, ResidualBlock
└── distribution.py           symlog, symexp, SimNorm, TwoHotDist

lucidwm/                    Layer 2: Algorithm packages
├── wm/                       World Models (Ha & Schmidhuber)
│   ├── model.py                VAE, MDNRNN, Controller
│   ├── dataset.py              FrameDataset, SequenceDataset
│   ├── train.py                train_vae, train_rnn, train_controller_cmaes
│   ├── env.py                  collect_data, rollout_agent
│   └── main.py
├── dreamer/                  Dreamer V1
│   ├── model.py                RSSM, ConvEncoder, ConvDecoder, TanhNormalActor, DreamerModel
│   ├── loss.py                 world_model_loss, behavior_losses, lambda_return
│   ├── env.py                  make_env_fn, act, evaluate_agent, collect_batched_steps
│   └── main.py
├── dreamerv2/                Dreamer V2
│   ├── model.py                CategoricalRSSM, DreamerV2Model
│   ├── loss.py                 world_model_loss, behavior_losses, imagine_rollout
│   └── main.py
├── diamond/                  DIAMOND
│   ├── model.py                DiffusionWorldModel, RewardTerminationModel, ActorCritic, ConditionalUNet
│   ├── train.py                diffusion_loss, reward_termination_loss, actor_critic_loss
│   ├── env.py                  make_diamond_atari_env, collect_steps, evaluate_agent
│   └── main.py
├── tdmpc2/                   TD-MPC2
│   ├── model.py                TDMPC2Model, OfflineDataset
│   ├── planner.py              plan_mppi
│   ├── train.py                update, train_offline
│   └── main.py
├── pwm/                      PWM (dmcontrol + mujoco unified)
│   ├── model.py                PWMModel, PWMPolicy, PWMCriticEnsemble
│   ├── train.py                pretrain_world_model, policy_update, td_lambda_targets
│   └── main.py                 --dataset dmcontrol|mujoco
├── dinowm/                   DINO-WM
│   ├── model.py                DINOv2Encoder, TransitionViT, ActionEmbedding
│   ├── dataset.py              TrajectoryDataset, TrajSlicerDataset
│   ├── train.py                train_world_model
│   ├── planner.py              plan_cem
│   └── main.py
└── leworldmodel/             LeWM
    ├── encoder.py              ViT-Tiny encoder
    ├── predictor.py            LeWMPredictor (causal ViT)
    ├── loss.py                 SIGReg
    ├── planner.py              plan_cem
    ├── dataset.py              LeWMH5Dataset
    ├── train.py                train
    └── main.py

lucidwm_utils/              Layer 3: Shared infrastructure
├── envs.py                   make_env() for DMC, Atari, MuJoCo, MetaWorld
├── buffers.py                ReplayBuffer
├── datasets.py               MinariDatasetAdapter
├── metrics.py                evaluate()
├── misc.py                   get_device, set_seed, set_requires_grad
├── logger.py                 W&B + TensorBoard
└── video.py                  Video recording
```

**Layer 1** components are extracted when used by 3+ algorithms with minimal variation. No cross-dependencies.

**Layer 2** algorithm packages import from Layers 1 and 3, but never from each other (except `pwm/` which imports `OfflineDataset` from `tdmpc2/` for the dmcontrol dataset variant).

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
# World Models: 4-phase training on CarRacing
python -m lucidwm.wm.main --env-id CarRacing-v3 --seed 1

# Dreamer V1 on DMControl pixels
python -m lucidwm.dreamer.main --env-id walker-walk --seed 1 --track

# Dreamer V2 on Atari
python -m lucidwm.dreamerv2.main --env-id ALE/Pong-v5 --seed 1 --track

# DIAMOND on Atari
python -m lucidwm.diamond.main --env-id BreakoutNoFrameskip-v4 --seed 1 --track

# TD-MPC2: online training on DMControl
python -m lucidwm.tdmpc2.main --env-id walker-walk --seed 1 --track

# TD-MPC2: offline training on released datasets
python -m lucidwm.tdmpc2.main --offline --dataset mt30 --data-dir ./data

# PWM: learn policy via FoG through frozen world model (DMControl offline data)
python -m lucidwm.pwm.main --dataset dmcontrol --hf-dataset mt30 --env-id walker-walk

# PWM on MuJoCo via Minari
python -m lucidwm.pwm.main --dataset mujoco --minari-id mujoco/halfcheetah/medium-v0 --eval

# DINO-WM: train dynamics on frozen DINOv2 features, plan with CEM
python -m lucidwm.dinowm.main --data-dir ./data/pusht --action-dim 2

# LeWM: end-to-end JEPA from pixels
python -m lucidwm.leworldmodel.main --data-path ./data/pusht_expert_train.h5 --action-dim 2
```

## Fidelity

These are educational implementations, not exact ports of the original repos. Some algorithms are close to the published method, while others intentionally simplify data loading, training structure, or planning to keep each file readable and focused.

## What's Covered

**VAE + RNN** (World Models): The original. Learn a compressed visual model and a recurrent dynamics model separately, then optimize a tiny controller entirely inside the dream.

**Latent Imagination / RSSM** (Dreamer V1, V2): Learn a recurrent stochastic state-space model with reconstruction and reward prediction, then optimize an actor and value function through imagined latent rollouts. V1 uses Gaussian RSSM on DMControl. V2 uses categorical RSSM with straight-through gradients on Atari.

**Diffusion World Model** (DIAMOND): Predict future Atari frames directly in pixel space with an EDM-style denoiser, then train a recurrent policy and value function entirely inside imagined rollouts.

**Implicit / SimNorm** (TD-MPC2, PWM): No decoder, no reconstruction. Predict next latent, reward, and value directly. SimNorm prevents latent collapse. TD-MPC2 plans with MPPI at test time. PWM shows you can skip planning entirely and learn a better policy by backpropagating through frozen dynamics.

**Frozen Foundation Model** (DINO-WM): Don't learn perception at all. Freeze DINOv2, learn only dynamics on top of its patch features. Plan by optimizing action sequences to reach goal embeddings.

**Learned JEPA** (LeWM): Learn perception end-to-end but prevent collapse with SIGReg (a statistical test enforcing Gaussian latents). Two loss terms total.

## Contributing

1. Create a new folder in `lucidwm/` named after the algorithm
2. Split into `model.py`, `train.py`, `main.py` at minimum; add `env.py`, `dataset.py`, `planner.py` as needed
3. Use Layer 1 components where applicable; add to `lucidwm_components/` if used by 3+ algorithms
4. Benchmark on ≥3 environments with ≥3 seeds
5. Write a companion doc in `docs/`

Quality bar: results within 1 std of published, `ruff` clean.

## License

MIT
