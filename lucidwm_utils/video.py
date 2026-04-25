"""Video recording and saving utilities."""

from __future__ import annotations

import numpy as np
from pathlib import Path

def save_video(
    frames: list[np.ndarray],
    path: str | Path,
    fps: int = 30,
):
    """Save a list of RGB frames as an MP4 video.

    Args:
        frames: List of (H, W, 3) uint8 arrays.
        path: Output file path (e.g., "videos/eval_step_1000.mp4").
        fps: Frames per second.
    """
    import imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps)

def save_video_grid(
    videos: list[list[np.ndarray]],
    path: str | Path,
    fps: int = 30,
    ncols: int = 4,
):
    """Save multiple videos as a grid.

    Args:
        videos: List of frame lists. Each inner list is one video.
        path: Output path.
        fps: Frames per second.
        ncols: Number of columns in grid.
    """
    import imageio

    # Pad to same length
    max_len = max(len(v) for v in videos)
    padded = []
    for v in videos:
        if len(v) < max_len:
            v = v + [v[-1]] * (max_len - len(v))
        padded.append(v)

    # Arrange in grid
    nrows = (len(padded) + ncols - 1) // ncols
    H, W = padded[0][0].shape[:2]

    grid_frames = []
    for t in range(max_len):
        grid = np.zeros((nrows * H, ncols * W, 3), dtype=np.uint8)
        for i, v in enumerate(padded):
            r, c = i // ncols, i % ncols
            grid[r * H : (r + 1) * H, c * W : (c + 1) * W] = v[t]
        grid_frames.append(grid)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), grid_frames, fps=fps)