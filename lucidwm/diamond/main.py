"""DIAMOND: Entry point

Usage:
  python -m lucidwm.diamond.main --env-id BreakoutNoFrameskip-v4 --seed 1
"""
from __future__ import annotations
import argparse
import torch
import torch.optim as optim
from lucidwm_utils.buffers import ReplayBuffer
from lucidwm_utils.logger import Logger
from lucidwm_utils.misc import get_device, set_seed
from .model import DiffusionWorldModel, RewardTerminationModel, ActorCritic
from .train import diffusion_loss, reward_termination_loss, actor_critic_loss, tensor_batch
from .env import make_diamond_atari_env, evaluate_agent, collect_steps

def parse_args():
    parser = argparse.ArgumentParser(description="LucidWM: DIAMOND on Atari")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lucidwm")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--prefill-steps", type=int, default=5_000)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--updates-per-epoch", type=int, default=400)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-len", type=int, default=4)
    parser.add_argument("--burnin-len", type=int, default=4)
    parser.add_argument("--imagine-horizon", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay-diffusion", type=float, default=1e-2)
    parser.add_argument("--weight-decay-rt", type=float, default=1e-2)
    parser.add_argument("--weight-decay-ac", type=float, default=0.0)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--diffusion-channels", type=int, default=64)
    parser.add_argument("--diffusion-cond-dim", type=int, default=256)
    parser.add_argument("--rt-channels", type=int, default=32)
    parser.add_argument("--rt-cond-dim", type=int, default=128)
    parser.add_argument("--lstm-dim", type=int, default=512)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--sigma-min", type=float, default=2e-3)
    parser.add_argument("--sigma-max", type=float, default=5.0)
    parser.add_argument("--sigma-rho", type=float, default=7.0)
    parser.add_argument("--p-mean", type=float, default=-0.4)
    parser.add_argument("--p-std", type=float, default=1.2)
    parser.add_argument("--sample-steps", type=int, default=3)
    parser.add_argument("--discount", type=float, default=0.985)
    parser.add_argument("--lambda-return", type=float, default=0.95)
    parser.add_argument("--entropy-weight", type=float, default=1e-3)
    parser.add_argument("--collect-eps", type=float, default=0.01)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    env = make_diamond_atari_env(args.env_id, args.seed, args.img_size)
    obs_shape = tuple(env.observation_space.shape)
    action_dim = int(env.action_space.n)

    diffusion_model = DiffusionWorldModel(
        action_dim=action_dim,
        context_len=args.context_len,
        base_channels=args.diffusion_channels,
        cond_dim=args.diffusion_cond_dim,
        sigma_data=args.sigma_data,
    ).to(device)
    rt_model = RewardTerminationModel(
        action_dim=action_dim,
        channels=args.rt_channels,
        cond_dim=args.rt_cond_dim,
        lstm_dim=args.lstm_dim,
    ).to(device)
    actor_critic = ActorCritic(action_dim=action_dim, lstm_dim=args.lstm_dim).to(device)

    diffusion_opt = optim.AdamW(diffusion_model.parameters(), lr=args.lr, eps=args.adam_eps, weight_decay=args.weight_decay_diffusion)
    rt_opt = optim.AdamW(rt_model.parameters(), lr=args.lr, eps=args.adam_eps, weight_decay=args.weight_decay_rt)
    ac_opt = optim.AdamW(actor_critic.parameters(), lr=args.lr, eps=args.adam_eps, weight_decay=args.weight_decay_ac)

    buffer = ReplayBuffer(
        capacity=args.buffer_size,
        obs_shape=obs_shape,
        action_dim=action_dim,
        seq_len=args.context_len + args.imagine_horizon,
    )
    logger = Logger(
        project=args.wandb_project,
        name=f"diamond_{args.env_id}_s{args.seed}",
        config=vars(args), use_wandb=args.track,
    )

    print(f"DIAMOND Atari | env={args.env_id} | obs={obs_shape} | act={action_dim} | device={device}")
    print("Collecting prefill experience...")
    prefill_metrics = collect_steps(env, actor_critic, buffer, args.prefill_steps, action_dim, device, deterministic=False, epsilon=1.0, seed=args.seed)
    total_steps = args.prefill_steps
    for ret, length in prefill_metrics:
        logger.log({"charts/episodic_return": ret, "charts/episodic_length": length}, step=total_steps)

    seq_len = args.context_len + args.imagine_horizon
    epoch = 0
    while total_steps < args.total_steps:
        epoch += 1
        episodic = collect_steps(env, actor_critic, buffer, args.steps_per_epoch, action_dim, device, deterministic=False, epsilon=args.collect_eps)
        total_steps += args.steps_per_epoch
        for ret, length in episodic:
            logger.log({"charts/episodic_return": ret, "charts/episodic_length": length}, step=total_steps)
        if buffer.num_episodes < 1:
            continue
        diff_metrics = {}
        rt_metrics = {}
        ac_metrics = {}
        for _ in range(args.updates_per_epoch):
            try:
                batch_np = buffer.sample(args.batch_size, seq_len=seq_len)
            except ValueError:
                break
            batch = tensor_batch(batch_np, device)
            diffusion_opt.zero_grad(set_to_none=True)
            diff_loss = diffusion_loss(diffusion_model, batch, args)
            diff_loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), args.grad_clip)
            diffusion_opt.step()
            diff_metrics = {"losses/diffusion": diff_loss.item()}
            rt_opt.zero_grad(set_to_none=True)
            rt_loss, rt_metrics = reward_termination_loss(rt_model, batch, args)
            rt_loss.backward()
            torch.nn.utils.clip_grad_norm_(rt_model.parameters(), args.grad_clip)
            rt_opt.step()
            ac_opt.zero_grad(set_to_none=True)
            policy_loss, value_loss, ac_metrics = actor_critic_loss(diffusion_model, rt_model, actor_critic, batch, args)
            ac_total = policy_loss + value_loss
            ac_total.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), args.grad_clip)
            ac_opt.step()
        metrics = {}
        metrics.update(diff_metrics)
        metrics.update(rt_metrics)
        metrics.update(ac_metrics)
        if metrics:
            logger.log(metrics, step=total_steps)
            print(
                f"epoch={epoch:04d} steps={total_steps:06d} "
                f"diff={metrics.get('losses/diffusion', 0.0):.4f} "
                f"reward={metrics.get('losses/reward_model', 0.0):.4f} "
                f"policy={metrics.get('losses/policy', 0.0):.4f} "
                f"value={metrics.get('losses/value', 0.0):.4f}"
            )
        if total_steps % args.eval_freq == 0:
            actor_critic.eval()
            eval_result = evaluate_agent(actor_critic, args.env_id, args.eval_episodes, args.img_size, args.seed, device)
            logger.log({
                "charts/eval_return": eval_result["mean_return"],
                "charts/eval_std": eval_result["std_return"],
                "charts/eval_length": eval_result["mean_length"],
            }, step=total_steps)
            print(f"eval steps={total_steps:06d} return={eval_result['mean_return']:.1f} +/- {eval_result['std_return']:.1f}")
            actor_critic.train()

    torch.save({
        "diffusion_model": diffusion_model.state_dict(),
        "reward_termination_model": rt_model.state_dict(),
        "actor_critic": actor_critic.state_dict(),
        "args": vars(args),
    }, f"diamond_{args.env_id.replace('/', '_')}_s{args.seed}.pt")
    logger.close()
    env.close()
