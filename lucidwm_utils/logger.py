"""Logging helpers for W&B and TensorBoard.

Provides a unified Logger that writes to both backends.
"""

from __future__ import annotations

from pathlib import Path

class Logger:
    """Unified logger for W&B and TensorBoard.

    Args:
        project: W&B project name.
        name: Run name.
        config: Hyperparameter dict to log.
        use_wandb: Whether to log to W&B.
        use_tb: Whether to log to TensorBoard.
        log_dir: TensorBoard log directory.
    """

    def __init__(
        self,
        project: str = "lucidwm",
        name: str = "",
        config: dict | None = None,
        use_wandb: bool = False,
        use_tb: bool = True,
        log_dir: str = "runs",
    ):
        self._wandb_run = None
        self._tb_writer = None

        if use_wandb:
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=project,
                    name=name,
                    config=config,
                    save_code=True,
                )
            except ImportError:
                print("wandb not installed, skipping W&B logging")

        if use_tb:
            from torch.utils.tensorboard import SummaryWriter

            run_dir = Path(log_dir) / (name or "default")
            self._tb_writer = SummaryWriter(str(run_dir))

            if config:
                # Log config as text
                config_str = "\n".join(f"{k}: {v}" for k, v in config.items())
                self._tb_writer.add_text("config", config_str)

    def log(self, metrics: dict[str, float], step: int):
        """Log scalar metrics.

        Args:
            metrics: Dict of metric_name -> value.
                     Use "/" for grouping: "losses/kl_loss", "charts/return".
            step: Global step count.
        """
        if self._wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step)

        if self._tb_writer is not None:
            for key, value in metrics.items():
                self._tb_writer.add_scalar(key, value, step)

    def log_video(self, key: str, frames, step: int, fps: int = 30):
        """Log video frames.

        Args:
            key: Metric name.
            frames: np.ndarray of shape (T, H, W, C) or (T, C, H, W).
            step: Global step.
            fps: Frames per second.
        """
        import numpy as np

        frames = np.array(frames)

        if self._wandb_run is not None:
            import wandb

            # wandb expects (T, H, W, C)
            if frames.shape[1] in (1, 3):  # (T, C, H, W)
                frames_hwc = frames.transpose(0, 2, 3, 1)
            else:
                frames_hwc = frames
            wandb.log({key: wandb.Video(frames_hwc, fps=fps)}, step=step)

        if self._tb_writer is not None:
            # tensorboard expects (1, T, C, H, W)
            if frames.shape[-1] in (1, 3):  # (T, H, W, C)
                frames_chw = frames.transpose(0, 3, 1, 2)
            else:
                frames_chw = frames
            self._tb_writer.add_video(
                key, frames_chw[None], step, fps=fps
            )

    def close(self):
        if self._wandb_run is not None:
            import wandb

            wandb.finish()
        if self._tb_writer is not None:
            self._tb_writer.close()
