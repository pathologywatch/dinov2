import glob
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(self, data_dir, transform=None, use_sample=False, subsample_patches=None):
        """
        Args:
            data_dir (string): Directory with images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.wsis = glob.glob(f'{data_dir}/*.json')
        self.wsis = [Path(wsi).stem for wsi in self.wsis]
        self.wsis = self.wsis[:100]
        self.patches_df = pd.DataFrame()
        for idx in tqdm(range(len(self.wsis))):
            wsi = self.wsis[idx]
            patches_df_path = f'{data_dir}/{wsi}_patches_df.parquet'
            tmp = pd.read_parquet(patches_df_path)
            if subsample_patches is not None:
                n_samples = min(len(tmp), subsample_patches)
                tmp = tmp.sample(n=n_samples)
            tmp['wsi'] = wsi
            self.patches_df = pd.concat([self.patches_df, tmp])
        print(f'Found {len(self.patches_df)} images')
        if use_sample:
            self.patches_df = self.patches_df.sample(1000)
            self.wsis = list(set(self.patches_df.wsi.values))
        if len(self.patches_df) == 0:
            raise ValueError(f'No patches found in {self.data_dir}!')

    def __len__(self):
        return len(self.patches_df)

    def __getitem__(self, idx):
        rel_path = self.patches_df.index.values[idx]
        path = f'{self.data_dir}/{rel_path}'
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, None
