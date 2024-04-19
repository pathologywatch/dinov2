import glob
import pandas as pd
from pathlib import Path
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

        wsis = glob.glob(f"{patches_dir}/*.json")
        for file in tqdm(wsis, desc="Reading data"):
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
