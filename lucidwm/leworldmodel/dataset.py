"""LeWorldModel: Dataset"""
import os
import subprocess
import numpy as np
import torch
from torch.utils.data import Dataset

def download_h5_dataset(repo_id, filename, cache_dir="data"):
    """Download and decompress the LeWM dataset from HuggingFace."""
    from huggingface_hub import hf_hub_download
    os.makedirs(cache_dir, exist_ok=True)
    h5_name = filename[:-4] if filename.endswith(".zst") else filename
    h5_path = os.path.join(cache_dir, h5_name)
    if os.path.exists(h5_path):
        print(f"Dataset already cached: {h5_path}")
        return h5_path
    print(f"Downloading {filename} from {repo_id}...")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=os.path.join(cache_dir, ".hf_cache"),
    )
    if filename.endswith(".zst"):
        print(f"Decompressing -> {h5_path}")
        try:
            subprocess.run(["zstd", "-d", downloaded, "-o", h5_path], check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            with open(downloaded, "rb") as ifh, open(h5_path, "wb") as ofh:
                dctx.copy_stream(ifh, ofh)
        print(f"Decompressed: {h5_path}")
    else:
        import shutil
        shutil.copy2(downloaded, h5_path)
    return h5_path

class LeWMH5Dataset(Dataset):
    """Dataset loader that returns LeWM history windows."""

    def __init__(self, h5_path: str, img_size: int = 224, frameskip: int = 1, history_size: int = 3):
        self.ensure_hdf5plugin()
        import h5py
        self.h5_path = h5_path
        self.img_size = img_size
        self.frameskip = frameskip
        self.history_size = history_size
        with h5py.File(h5_path, "r") as f:
            self.ep_len = f["ep_len"][:]
            self.ep_offset = f["ep_offset"][:]
            self.actions = f["action"][:]
            pixels_shape = f["pixels"].shape
            self.stored_h, self.stored_w = pixels_shape[1], pixels_shape[2]
            print(f"Loaded H5: {h5_path}")
            print(f"  Episodes: {len(self.ep_len)}, Total steps: {pixels_shape[0]}")
            print(f"  Pixels: {pixels_shape[1:]}, Actions: {self.actions.shape[1:]}")
        self.index = []
        stride = self.frameskip
        max_offset = self.history_size * stride
        for ep_idx in range(len(self.ep_len)):
            offset = int(self.ep_offset[ep_idx])
            length = int(self.ep_len[ep_idx])
            for t in range(length - max_offset):
                self.index.append(offset + t)
        print(f"  Using all {len(self.ep_len)} episodes, {len(self.index)} history windows")
        self.h5 = None

    def get_h5(self):
        if self.h5 is None:
            self.ensure_hdf5plugin()
            import h5py
            self.h5 = h5py.File(self.h5_path, "r")
        return self.h5

    def open_h5(self):
        self.ensure_hdf5plugin()
        import h5py
        if self.h5 is not None:
            try:
                self.h5.close()
            except Exception:
                pass
        self.h5 = h5py.File(self.h5_path, "r")

    @staticmethod
    def ensure_hdf5plugin():
        os.environ.setdefault("HDF5_PLUGIN_PATH", "")
        try:
            import hdf5plugin  # noqa: F401
        except ImportError:
            pass

    @staticmethod
    def worker_init_fn(worker_id):
        os.environ.setdefault("HDF5_PLUGIN_PATH", "")
        try:
            import hdf5plugin  # noqa: F401
        except ImportError:
            pass
        import torch.utils.data as data
        worker_info = data.get_worker_info()
        if worker_info is not None:
            dataset = worker_info.dataset
            if isinstance(dataset, LeWMH5Dataset):
                dataset.open_h5()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        f = self.get_h5()
        global_t = self.index[idx]
        step = self.frameskip
        frame_ids = [global_t + i * step for i in range(self.history_size + 1)]
        action_ids = [global_t + i * step for i in range(self.history_size)]
        obs = torch.stack([self.preprocess(f["pixels"][fid]) for fid in frame_ids], dim=0)
        action = torch.from_numpy(self.actions[action_ids].copy()).float()
        return {"obs": obs, "action": action}

    def preprocess(self, img):
        import cv2
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
