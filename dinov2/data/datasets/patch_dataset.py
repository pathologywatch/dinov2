import glob
import json
import os
from pathlib import Path

from openslide import OpenSlide
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(self, patches_dir, transform=None):
        """
        Args:
            patches_dir (string): Directory with patch images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.patches_dir = patches_dir
        self.transform = transform
        self.patches = []

        config_files = glob.glob(f"{patches_dir}/*.json")
        self.wsis = [Path(c).stem for c in config_files]
        for file in tqdm(config_files, desc="Reading data"):
            with open(file, "r") as f:
                slide_data = json.load(f)
            for patch in slide_data["patch_pos"]:
                patch_file = patch.get("patch_file")
                # If patches were generated
                if patch_file is not None:
                    self.patches.append(
                        {
                            "on_disk": True,
                            "patch_path": os.path.join(patches_dir, patch_file),
                        }
                    )
                else:
                    p_x1 = patch["x1"]
                    p_y1 = patch["y1"]
                    p_width = patch["x2"] - p_x1
                    p_height = patch["y2"] - p_y1
                    self.patches.append(
                        {
                            "on_disk": False,
                            "slide_path": slide_data["slide_path"],
                            "slide_level": slide_data["slide_level"],
                            "patch_pos": (p_x1, p_y1, p_width, p_height),
                        }
                    )
        print(f"Dataset will use {len(self)} patches")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_data = self.patches[idx]
        if patch_data["on_disk"]:
            patch_img = Image.open(patch_data["patch_path"]).convert("RGB")
        else:
            level = patch_data["slide_level"]
            with OpenSlide(patch_data["slide_path"]) as osr:
                # Reads the entire slide
                x1, y1, width, height = patch_data["patch_pos"]
                patch_img = osr.read_region((x1, y1), level, (width, height))
            patch_img = patch_img.convert("RGB")
        if self.transform:
            patch_img = self.transform(patch_img)
        return patch_img, None
