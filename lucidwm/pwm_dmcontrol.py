"""LucidWM: PWM (Policy Learning with Multi-Task World Models) on DMControl

Paper:     "PWM: Policy Learning with Multi-Task World Models" (Georgiev et al., ICLR 2025)
Reference: https://github.com/imgeorgiev/PWM

Core insight:
  Well-regularized world models create SMOOTHER optimization landscapes than
  the true dynamics. This means first-order gradient (FoG) optimization through
  the world model produces better policies than zeroth-order methods (MPPI/CEM)
  and even better than FoG through the ground-truth simulator.

  Key finding: accuracy of the world model is INVERSELY correlated with policy
  performance. Smooth > accurate for gradient-based policy learning.

Architecture:
  World model:  Same as TD-MPC2 (encoder + dynamics + reward, all with SimNorm).
                Pre-trained on offline data, then FROZEN during policy learning.
  Policy:       MLP, z -> (mu, sigma), squashed Gaussian. Trained via FoG
                backpropagated THROUGH the frozen world model dynamics.
  Critic:       Ensemble of 3 value MLPs, trained via TD(lambda) in latent space.
                NOT via discrete regression bins -- uses standard MSE.

Training (2 phases):
  1. Pre-train world model on offline data (same as TD-MPC2, Eq. 10)
     - Consistency loss: ||F(z,a) - sg(E(s'))||^2
     - Reward loss: CE on discrete bins
     - Auto-regressive, horizon H=16, discount gamma=0.99
  2. Learn policy via FoG through frozen world model (<10 min per task)
     - Actor loss: backprop reward through dynamics chain (Eq. 6)
     - Critic loss: TD(lambda) over H-step imagined rollout (Eq. 7-9)
     - Batched: multiple trajectories imagined in parallel

Key hyperparameters (from paper Appendix C):
  - World model: same as TD-MPC2 (512 latent, 512 hidden, SimNorm, Mish+LN)
  - Policy horizon: H=5 (single task), H=16 (multi-task)
  - Policy lr: 1e-3
  - Critic lr: 1e-3
  - Critic ensemble: 3
  - Lambda (TD): 0.95
  - Discount: 0.99
  - Policy batch size: 32 (small batches work better for FoG)
  - Policy training steps: 10k-50k (~10 min per task)

Components: Reuses TD-MPC2 world model architecture (SimNorm, discrete regression)
Data:       Uses TD-MPC2 offline datasets (nicklashansen/tdmpc2 on HuggingFace)
            OR pre-trained TD-MPC2 world model checkpoints (imgeorgiev/pwm on HuggingFace)
Env:        DMControl, MetaWorld (MT30/MT80 benchmarks)
Target:     27% higher reward than TD-MPC2 on MT80 without online planning
Compute:    <10 minutes per task for policy learning (on RTX 6000)
"""

import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lucidwm_utils.envs import make_env
from lucidwm_utils.metrics import evaluate
from lucidwm_utils.logger import Logger
from lucidwm_utils.misc import set_seed, get_device

from lucidwm_components.networks import MLP
from lucidwm_components.distribution import TwoHotDist, SimNorm

def parse_args():
    parser = argparse.ArgumentParser(description="LucidWM: PWM")

    # Standard args
    parser.add_argument("--env-id", type=str, default="walker-walk")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lucidwm")
    parser.add_argument("--eval-freq", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--action-repeat", type=int, default=2)

    # World model pre-training
    parser.add_argument("--wm-checkpoint", type=str, default=None,
                        help="Path to pre-trained world model checkpoint. "
                             "If None, trains from scratch on offline data.")
    parser.add_argument("--wm-lr", type=float, default=3e-4)
    parser.add_argument("--wm-epochs", type=int, default=100)
    parser.add_argument("--wm-batch-size", type=int, default=256)
    parser.add_argument("--wm-horizon", type=int, default=16,
                        help="Training horizon for world model (16 for better FoG gradients)")

    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-bins", type=int, default=101)
    parser.add_argument("--simnorm-dim", type=int, default=8)

    # Policy learning (phase 2)
    parser.add_argument("--policy-steps", type=int, default=10_000,
                        help="Gradient steps for policy learning (<10 min)")
    parser.add_argument("--policy-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--policy-batch-size", type=int, default=32,
                        help="Small batches work better for FoG (paper finding)")
    parser.add_argument("--policy-horizon", type=int, default=5,
                        help="Imagination horizon for policy training")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--lmbd", type=float, default=0.95, help="TD(lambda)")
    parser.add_argument("--num-critics", type=int, default=3)

    # Data
    parser.add_argument("--dataset", type=str, default="mt30", choices=["mt30", "mt80"])
    parser.add_argument("--data-dir", type=str, default="data/tdmpc2")

    return parser.parse_args()

# PWM World Model
class PWMModel(nn.Module):
    """TD-MPC2-style implicit world model used as a differentiable simulator.

    Components:
      encoder:  obs -> z (+ SimNorm)
      dynamics: (z, a) -> z' (+ SimNorm)  -- DIFFERENTIABLE, gradients flow through
      reward:   (z, a) -> bins (discrete regression)

    In PWM, this model is pre-trained then FROZEN. Policy gradients flow
    through dynamics via backprop, which is the key innovation.
    """

    def __init__(self, obs_dim: int, action_dim: int, args):
        super(PWMModel, self).__init__()

        self.encoder = nn.Sequential(MLP(obs_dim, args.latent_dim, args.hidden_dim, activation=nn.Mish), SimNorm(args.simnorm_dim))
        self.dynamics = nn.Sequential(MLP(args.latent_dim+action_dim, self.latent_dim, args.hidden_dim, activation=nn.Mish), SimNorm(args.simnorm_dim))
        self.rewards = MLP(args.latent_dim+action_dim, args.num_bins, args.hidden_dim, activation=nn.Mish)
        self.two_hot_distribution = TwoHotDist(num_bins=args.num_bins)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, obs):
        """obs (B, O) -> z (B, LATENT_DIM)."""
        return self.encoder(obs)

    def next_state(self, z, a):
        """(z, a) -> z'. Differentiable -- gradients flow through for FoG."""
        return self.dynamics(torch.cat([z, a], dim=-1))

    def reward(self, z, a):
        """(z, a) -> reward logits (B, NUM_BINS)."""
        return self.rewards(torch.cat([z, a], dim=-1))
    
    def reward_encode(self, x):
        return self.two_hot_distribution.encode(x)

    def reward_scalar(self, z, a):
        """(z, a) -> scalar reward (B,). Differentiable through softmax."""
        return self.two_hot_distribution.decode(self.reward(z, a))
    
class PWMPolicy(nn.Module):
    """Squashed Gaussian policy: z -> (mu, log_std) -> tanh(sample).

    Same architecture as TD-MPC2's policy but trained via FoG
    through the frozen world model, not via DDPG-style Q-gradients.
    """

    def __init__(self, args, action_dim):
        super().__init__()
        self.net = MLP(args.latent_dim, 2 * action_dim, activation=nn.Mish)

    def forward(self, z, deterministic=False):
        """z (B, LATENT) -> action (B, A), log_prob (B,)."""
        out = self.net(z)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        std = log_std.exp()

        if deterministic:
            return torch.tanh(mean), torch.zeros(z.shape[0], device=z.device)

        noise = torch.randn_like(mean)
        raw = mean + std * noise
        action = torch.tanh(raw)

        # Log prob with tanh correction
        log_prob = (-0.5 * noise.pow(2) - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        log_prob -= (2 * (np.log(2) - raw - F.softplus(-2 * raw))).sum(-1)

        return action, log_prob

class PWMCriticEnsemble(nn.Module):
    """Ensemble of value function critics V(z).

    Note: PWM uses V(z), NOT Q(z,a). The critic estimates the value of
    a latent state, and the actor is trained by backpropagating reward
    through the dynamics, not through the critic.

    Ensemble of 3 reduces variance (paper Section 3.3).
    """

    def __init__(self, args, num_critics=3):
        super().__init__()
        self.critics = nn.ModuleList([
            MLP(args.latent_dim, 1, activation=nn.Mish) for _ in range(num_critics)
        ])

    def forward(self, z):
        """z (B, LATENT) -> values list of (B,) tensors."""
        return [critic(z).squeeze(-1) for critic in self.critics]

    def mean_value(self, z):
        """z (B, LATENT) -> mean value (B,)."""
        values = self.forward(z)
        return torch.stack(values).mean(dim=0)
    
# Training
def pretrain_world_model(world_model, dataset, args, device, logger):
    """Pre-train world model on offline data (same loss as TD-MPC2, Eq. 10).

    Loss = sum over horizon of:
      gamma^t * (consistency_loss + reward_cross_entropy)

    Where consistency_loss = ||F(z_t, a_t) - sg(E(s_{t+1}))||^2
    and reward is discrete regression via cross-entropy on bins.

    Uses H=16 for training horizon (better FoG gradients, Section 3.2).
    """
    optimizer = optim.Adam(world_model.parameters(), lr=args.wm_lr)
    H = args.wm_horizon
    seq_len = H + 1

    print(f"Pre-training world model: {args.wm_epochs} epochs, H={H}")

    step = 0
    for epoch in range(args.wm_epochs):
        world_model.train()
        epoch_loss = 0.0
        n_batches = max(1, dataset.total_transitions // (args.wm_batch_size * seq_len))
        n_batches = min(n_batches, 1000)  # cap per epoch

        for _ in range(n_batches):
            batch = dataset.sample(args.wm_batch_size, seq_len=seq_len)
            obs = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
            action = torch.tensor(batch["action"], dtype=torch.float32, device=device)
            reward = torch.tensor(batch["reward"], dtype=torch.float32, device=device)

            z = world_model.encode(obs[:, 0])
            total_loss = torch.tensor(0.0, device=device)

            for t in range(H):
                # Consistency: predicted next latent vs. encoded true next obs
                z_pred = world_model.next_state(z, action[:, t])
                with torch.no_grad():
                    z_target = world_model.encode(obs[:, t + 1])
                consistency = F.mse_loss(z_pred, z_target)

                # Reward: discrete regression cross-entropy
                r_logits = world_model.reward(z, action[:, t])
                r_target = world_model.reward_encode(reward[:, t])
                reward_loss = -(r_target * F.log_softmax(r_logits, dim=-1)).sum(-1).mean()

                total_loss = total_loss + (args.discount ** t) * (consistency + reward_loss)
                z = z_pred  # auto-regressive

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(world_model.parameters(), 20.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            step += 1

        avg = epoch_loss / n_batches
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{args.wm_epochs}  loss={avg:.4f}")
        logger.log({"wm/loss": avg}, step=step)

    return world_model

# Policy training
def td_lambda_targets(rewards, values, discount, lmbd):
    """Compute TD(lambda) targets for imagined rollout.

    Args:
        rewards: (B, H) imagined rewards
        values:  (B, H) critic value estimates
        discount: scalar
        lmbd: scalar

    Returns:
        targets: (B, H) TD(lambda) targets
    """
    B, H = rewards.shape
    targets = torch.zeros_like(rewards)
    next_value = values[:, -1]

    for t in reversed(range(H)):
        if t == H - 1:
            next_target = next_value
        else:
            next_target = (1 - lmbd) * values[:, t + 1] + lmbd * targets[:, t + 1]
        targets[:, t] = rewards[:, t] + discount * next_target

    return targets

def policy_update(
    world_model: PWMModel,
    policy: PWMPolicy,
    critic: PWMCriticEnsemble,
    policy_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    start_states: torch.Tensor,
    args,
) -> dict[str, float]:
    """Single policy learning step via FoG through frozen world model.

    This is the core of PWM (Algorithm 1, Eq. 6-9):

    1. Sample initial states s1 from data
    2. Encode to z1 = E(s1) (no grad through encoder)
    3. Rollout H steps through world model using policy:
       for h in [1..H]:
         a_h ~ pi(z_h)
         r_h = R(z_h, a_h)      <- differentiable
         z_{h+1} = F(z_h, a_h)  <- differentiable, THIS IS THE KEY
    4. Actor loss: maximize sum of gamma^h * r_h + gamma^H * V(z_H)
       Gradients flow: loss -> rewards -> dynamics -> policy
    5. Critic loss: TD(lambda) on the imagined trajectory

    Args:
        world_model: FROZEN world model (no gradients to its parameters)
        policy: actor network being trained
        critic: value ensemble being trained
        policy_optimizer: Adam for policy
        critic_optimizer: Adam for critic
        start_states: (B, obs_dim) initial observations sampled from dataset
        args: hyperparameters

    Returns:
        metrics dict
    """
    H = args.policy_horizon
    gamma = args.discount

    # Encode start states (detach from world model graph)
    with torch.no_grad():
        z = world_model.encode(start_states)  # (B, LATENT)

    # ── Imagine rollout ───────────────────────────────────────────────
    # Collect rewards and values along the trajectory.
    # Gradients flow through dynamics and reward for actor update.
    imagined_rewards = []
    imagined_values = []
    imagined_log_probs = []
    z_current = z

    for h in range(H):
        # Value estimate (detached for actor loss, used for critic targets)
        with torch.no_grad():
            v = critic.mean_value(z_current)
        imagined_values.append(v)

        # Sample action from policy
        action, log_prob = policy(z_current)
        imagined_log_probs.append(log_prob)

        # Get reward (differentiable through world model)
        reward = world_model.reward_scalar(z_current, action)  # (B,)
        imagined_rewards.append(reward)

        # Step dynamics (differentiable -- FoG gradients flow here)
        z_current = world_model.next_state(z_current, action)  # (B, LATENT)

    # Terminal value
    with torch.no_grad():
        terminal_value = critic.mean_value(z_current)  # (B,)

    rewards_tensor = torch.stack(imagined_rewards, dim=1)  # (B, H)
    values_tensor = torch.stack(imagined_values, dim=1)    # (B, H)

    # ── Actor loss (Eq. 6): maximize discounted rewards + terminal value ──
    # This is the FoG loss -- gradients backprop through reward_scalar and
    # next_state into the policy parameters.
    discounts = gamma ** torch.arange(H, device=z.device, dtype=torch.float32)
    actor_loss = -(rewards_tensor * discounts).sum(dim=1).mean()
    actor_loss -= (gamma ** H) * terminal_value.mean()

    policy_optimizer.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    policy_optimizer.step()

    # ── Critic loss (Eq. 7-9): TD(lambda) on imagined trajectory ──────
    with torch.no_grad():
        # Re-imagine with updated policy for fresh targets
        z_c = world_model.encode(start_states)
        rew_for_critic = []
        for h in range(H):
            a, _ = policy(z_c)
            r = world_model.reward_scalar(z_c, a)
            rew_for_critic.append(r)
            z_c = world_model.next_state(z_c, a)
        rew_critic = torch.stack(rew_for_critic, dim=1)  # (B, H)
        terminal_v = critic.mean_value(z_c)

    # Get current value predictions
    z_c2 = world_model.encode(start_states).detach()
    val_preds = []
    for h in range(H):
        with torch.no_grad():
            a, _ = policy(z_c2)
        v = critic.mean_value(z_c2)
        val_preds.append(v)
        with torch.no_grad():
            z_c2 = world_model.next_state(z_c2, a)

    val_tensor = torch.stack(val_preds, dim=1)  # (B, H)

    # TD(lambda) targets
    with torch.no_grad():
        # Append terminal value for bootstrap
        all_values = torch.cat([val_tensor.detach(), terminal_v.unsqueeze(1)], dim=1)
        targets = td_lambda_targets(
            rew_critic, all_values[:, :-1], gamma, args.lmbd
        )

    critic_loss = F.mse_loss(val_tensor, targets)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
    critic_optimizer.step()

    return {
        "losses/actor": actor_loss.item(),
        "losses/critic": critic_loss.item(),
        "algo/mean_imagined_reward": rewards_tensor.detach().mean().item(),
        "algo/mean_value": values_tensor.mean().item(),
    }

def load_offline_dataset(args):
    """Load offline dataset. Uses TD-MPC2's OfflineDataset class."""
    try:
        from lucidwm.tdmpc2_dmcontrol import OfflineDataset
        return OfflineDataset(
            dataset=args.dataset,
            data_dir=args.data_dir,
        )
    except ImportError:
        raise ImportError(
            "PWM requires the TD-MPC2 OfflineDataset. "
            "Make sure lucidwm.tdmpc2_dmc is available."
        )

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    logger = Logger(
        project=args.wandb_project,
        name=f"pwm_{args.env_id}_s{args.seed}",
        config=vars(args),
        use_wandb=args.track,
    )

    # ── Phase 1: World Model ─────────────────────────────────────────
    if args.wm_checkpoint:
        # Load pre-trained world model
        print(f"Loading world model from {args.wm_checkpoint}")
        ckpt = torch.load(args.wm_checkpoint, map_location=device, weights_only=False)

        # Infer dims from checkpoint
        if "args" in ckpt:
            obs_dim = ckpt["args"].get("obs_dim", None)
            action_dim = ckpt["args"].get("action_dim", None)
        else:
            obs_dim = None
            action_dim = None

        # Fallback: infer from env
        if obs_dim is None or action_dim is None:
            env = make_env(args.env_id, args.seed, action_repeat=args.action_repeat)
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            env.close()

        world_model = PWMModel(obs_dim, action_dim, args).to(device)

        # Try to load state dict (handle both direct and nested formats)
        if "model" in ckpt:
            # Try loading only world model keys from a TD-MPC2 checkpoint
            wm_state = {}
            for k, v in ckpt["model"].items():
                if any(k.startswith(prefix) for prefix in ["encoder", "dynamics", "reward_head"]):
                    wm_state[k] = v
            if wm_state:
                world_model.load_state_dict(wm_state, strict=False)
            else:
                world_model.load_state_dict(ckpt["model"], strict=False)
        elif "world_model" in ckpt:
            world_model.load_state_dict(ckpt["world_model"])
        else:
            world_model.load_state_dict(ckpt, strict=False)

        print(f"World model loaded: obs={obs_dim}, act={action_dim}")

    else:
        # Pre-train from scratch on offline data
        print("Pre-training world model on offline data...")
        dataset = load_offline_dataset(args)

        obs_dim = dataset.obs_all.shape[-1]
        action_dim = dataset.action_all.shape[-1]

        world_model = PWMModel(obs_dim, action_dim, args).to(device)
        two_hot_distribution = 
        world_model = pretrain_world_model(world_model, dataset, args, device, logger)

        # Save
        wm_path = f"pwm_wm_{args.dataset}_s{args.seed}.pt"
        torch.save({
            "world_model": world_model.state_dict(),
            "args": {"obs_dim": obs_dim, "action_dim": action_dim},
        }, wm_path)
        print(f"Saved world model to {wm_path}")
    
    # Freeze world model
    world_model.eval()
    world_model.requires_grad_(False)
    print(f"World model frozen ({sum(p.numel() for p in world_model.parameters())/1e6:.1f}M params)")

    # ── Phase 2: Policy Learning via FoG ─────────────────────────────
    if obs_dim is None:
        env = make_env(args.env_id, args.seed, action_repeat=args.action_repeat)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()

    policy = PWMPolicy(action_dim).to(device)
    critic = PWMCriticEnsemble(num_critics=args.num_critics).to(device)

    policy_opt = optim.Adam(policy.parameters(), lr=args.policy_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=args.critic_lr)

    # Load dataset for sampling start states (or use existing)
    if not args.wm_checkpoint:
        # dataset already loaded above
        pass
    else:
        dataset = load_offline_dataset(args)

    print(f"\nPhase 2: Policy learning via FoG")
    print(f"  Steps: {args.policy_steps}, batch: {args.policy_batch_size}, "
          f"horizon: {args.policy_horizon}")
    print(f"  Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Critic params: {sum(p.numel() for p in critic.parameters()):,}")

    for step in range(args.policy_steps):
        # Sample start states from offline data
        batch = dataset.sample(args.policy_batch_size, seq_len=1)
        start_obs = torch.tensor(batch["obs"][:, 0], dtype=torch.float32, device=device)

        metrics = policy_update(
            world_model, policy, critic,
            policy_opt, critic_opt,
            start_obs, args,
        )

        if step % 500 == 0:
            logger.log(metrics, step=step)
            print(f"  Step {step:>6d}/{args.policy_steps}  "
                  f"actor={metrics['losses/actor']:.4f}  "
                  f"critic={metrics['losses/critic']:.4f}  "
                  f"reward={metrics['algo/mean_imagined_reward']:.3f}")

        # Evaluate in real env
        if step > 0 and step % args.eval_freq == 0:
            policy.eval()

            def agent_fn(o):
                with torch.no_grad():
                    o_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                    z = world_model.encode(o_t)
                    a, _ = policy(z, deterministic=True)
                    return a.squeeze(0).cpu().numpy()

            try:
                eval_result = evaluate(
                    lambda: make_env(args.env_id, args.seed + 100,
                                     action_repeat=args.action_repeat),
                    agent_fn,
                    num_episodes=args.eval_episodes,
                )
                print(f"  EVAL step {step}: {eval_result['mean_return']:.1f} "
                      f"+/- {eval_result['std_return']:.1f}")
                logger.log({
                    "charts/eval_return": eval_result["mean_return"],
                    "charts/eval_std": eval_result["std_return"],
                }, step=step)
            except Exception as e:
                print(f"  Eval failed: {e}")

            policy.train()

    # Save final policy
    policy_path = f"pwm_policy_{args.env_id}_s{args.seed}.pt"
    torch.save({
        "policy": policy.state_dict(),
        "critic": critic.state_dict(),
        "args": vars(args),
    }, policy_path)
    print(f"\nSaved policy to {policy_path}")

    # Final evaluation
    policy.eval()

    def agent_fn(o):
        with torch.no_grad():
            o_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
            z = world_model.encode(o_t)
            a, _ = policy(z, deterministic=True)
            return a.squeeze(0).cpu().numpy()

    try:
        eval_result = evaluate(
            lambda: make_env(args.env_id, args.seed + 200, action_repeat=args.action_repeat),
            agent_fn,
            num_episodes=50,
        )
        print(f"\nFinal eval (50 episodes): {eval_result['mean_return']:.1f} "
              f"+/- {eval_result['std_return']:.1f}")
        logger.log({
            "charts/final_return": eval_result["mean_return"],
        }, step=args.policy_steps)
    except Exception as e:
        print(f"Final eval failed: {e}")

    logger.close()
    print("Done.")
