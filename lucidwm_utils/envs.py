"""Environment creation and wrapper stacking.

Standard wrapper stacks per environment suite, documented with citations.
Algorithms needing non-standard wrappers override in their single file.
"""

from __future__ import annotations


import numpy as np
import gymnasium as gym

# Wrappers
class ActionRepeatWrapper(gym.Wrapper):
    """Repeat each action for `repeat` steps, summing rewards.

    Used by DMControl experiments (default repeat=2 for Dreamer, TD-MPC2).
    """

    def __init__(self, env: gym.Env, repeat: int = 2):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

class ResizeObservation(gym.ObservationWrapper):
    """Resize image observations to (size, size)."""

    def __init__(self, env: gym.Env, size: int = 64):
        super().__init__(env)
        self.size = size
        shape = (size, size) + env.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

    def observation(self, obs):
        import cv2

        return cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)

class GrayscaleObservation(gym.ObservationWrapper):
    """Convert RGB observations to grayscale."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2] + (1,)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, obs):
        import cv2

        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return gray[:, :, np.newaxis]

class NormalizeActions(gym.ActionWrapper):
    """Normalize continuous actions to [-1, 1]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._low = env.action_space.low
        self._high = env.action_space.high
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32
        )

    def action(self, action):
        # Map [-1, 1] -> [low, high]
        return self._low + (action + 1.0) * 0.5 * (self._high - self._low)

class PixelObservation(gym.ObservationWrapper):
    """Render pixel observations from state-based envs (e.g., DMControl).

    Transposes from (H, W, C) to (C, H, W) and normalizes to [0, 1].
    """

    def __init__(self, env: gym.Env, size: int = 64):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3, size, size), dtype=np.float32
        )

    def observation(self, obs):
        # For DMC via shimmy, obs is already pixels if render_mode="rgb_array"
        if isinstance(obs, dict) and "pixels" in obs:
            pixels = obs["pixels"]
        elif isinstance(obs, np.ndarray) and obs.ndim == 3:
            pixels = obs
        else:
            pixels = self.env.render()

        import cv2

        pixels = cv2.resize(pixels, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return pixels.transpose(2, 0, 1).astype(np.float32) / 255.0

# Factory
def make_env(
    env_id: str,
    seed: int = 0,
    img_size: int = 64,
    action_repeat: int | None = None,
    frame_stack: int | None = None,
) -> gym.Env:
    """Create environment with standard wrapper stack.

    Wrapper stacks by suite:
        DMControl:  PixelObs(64) -> ActionRepeat(2) -> NormalizeActions
        Atari:      NoopReset -> MaxAndSkip(4) -> Resize(64) -> Grayscale -> FrameStack(4)
        Crafter:    Resize(64)
        MetaWorld:  NormalizeActions
        Gymnasium:  Raw (for CarRacing, etc.)

    Args:
        env_id: Environment identifier.
            DMControl: "domain-task" format, e.g., "walker-walk", "cheetah-run".
            Atari: Standard gym ID, e.g., "BreakoutNoFrameskip-v4".
            Crafter: "crafter-reward-v1".
            MetaWorld: Task name, e.g., "reach-v2".
        seed: Random seed.
        img_size: Image observation size. Default: 64.
        action_repeat: Override default action repeat. None = use suite default.
        frame_stack: Override default frame stack. None = use suite default.

    Returns:
        Wrapped gymnasium environment.
    """
    suite = _detect_suite(env_id)

    if suite == "dmc":
        env = _make_dmc(env_id, seed, img_size, action_repeat or 2)
    elif suite == "atari":
        env = _make_atari(env_id, seed, img_size, frame_stack or 4)
    elif suite == "crafter":
        env = _make_crafter(seed, img_size)
    elif suite == "metaworld":
        env = _make_metaworld(env_id, seed)
    else:
        # Generic gymnasium env (e.g., CarRacing)
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)

    return env

def _detect_suite(env_id: str) -> str:
    """Detect environment suite from env_id string."""
    dmc_domains = {
        "cartpole", "cheetah", "walker", "reacher", "cup", "finger",
        "hopper", "humanoid", "quadruped", "dog", "acrobot", "pendulum",
    }
    if "-" in env_id and env_id.split("-")[0] in dmc_domains:
        return "dmc"
    if "NoFrameskip" in env_id or "ALE/" in env_id:
        return "atari"
    if "crafter" in env_id.lower():
        return "crafter"
    if env_id.endswith("-v2") and any(
        kw in env_id.lower()
        for kw in ["reach", "push", "pick", "drawer", "door", "window", "button"]
    ):
        return "metaworld"
    return "gym"

def _make_dmc(env_id: str, seed: int, img_size: int, action_repeat: int) -> gym.Env:
    """Create DMControl environment with pixel observations."""
    domain, task = env_id.split("-", 1)

    # Use shimmy to wrap dm_control for gymnasium compatibility
    from shimmy import DmControlCompatibilityV0

    import dm_control.suite as suite

    dm_env = suite.load(domain, task, task_kwargs={"random": seed})
    env = DmControlCompatibilityV0(dm_env, render_mode="rgb_array")

    env = PixelObservation(env, size=img_size)
    env = ActionRepeatWrapper(env, repeat=action_repeat)
    env = NormalizeActions(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def _make_atari(env_id: str, seed: int, img_size: int, frame_stack: int) -> gym.Env:
    """Create Atari environment with standard preprocessing."""
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=img_size,
        grayscale_obs=True,
        scale_obs=True,
    )
    env = gym.wrappers.FrameStackObservation(env, frame_stack)
    return env

def _make_crafter(seed: int, img_size: int) -> gym.Env:
    """Create Crafter environment."""
    import crafter

    env = crafter.Env(seed=seed)
    # Crafter returns its own gym-like interface; wrap for gymnasium
    # This is a placeholder -- exact wrapping depends on crafter version
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def _make_metaworld(env_id: str, seed: int) -> gym.Env:
    """Create MetaWorld environment."""
    import metaworld

    mt1 = metaworld.MT1(env_id, seed=seed)
    env = mt1.train_classes[env_id]()
    task = mt1.train_tasks[0]
    env.set_task(task)
    env = NormalizeActions(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env
