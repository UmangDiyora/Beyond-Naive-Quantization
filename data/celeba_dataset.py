import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


class CelebADataset(Dataset):
    """
    Minimal CelebA dataset for this project, providing:
    - (image, label, sensitive) tuples
    - balanced calibration subset selection
    Expected structure under `root`:
      root/img_align_celeba/...
      root/list_attr_celeba.txt
      root/list_eval_partition.txt
    """
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        protected_attributes: Optional[List[str]] = None,
        target_attribute: str = 'Smiling',
        subsample: Optional[int] = None
    ):
        self.root = root
        self.split = split
        self.transform = transform or self._default_transform()
        self.protected_attributes = protected_attributes or ['Male', 'Young']
        self.target_attribute = target_attribute

        self.img_dir = os.path.join(root, 'img_align_celeba')
        self.attr_file_txt = os.path.join(root, 'list_attr_celeba.txt')
        self.attr_file_csv = os.path.join(root, 'list_attr_celeba.csv')
        self.split_file_txt = os.path.join(root, 'list_eval_partition.txt')
        self.split_file_csv = os.path.join(root, 'list_eval_partition.csv')

        self._load_attributes()
        self._load_split()

        if subsample and subsample < len(self.image_paths):
            indices = np.random.choice(len(self.image_paths), subsample, replace=False)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = self.labels[indices]
            self.sensitive_features = self.sensitive_features[indices]

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_attributes(self):
        # TXT format (official)
        if os.path.exists(self.attr_file_txt):
            with open(self.attr_file_txt, 'r') as f:
                lines = f.readlines()
            self.all_attr_names = lines[1].strip().split()
            self.attr_data = {}
            for line in lines[2:]:
                parts = line.strip().split()
                img_name = parts[0]
                attrs = [int(x) for x in parts[1:]]
                attrs = [(x + 1) // 2 for x in attrs]  # map {-1,1} -> {0,1}
                self.attr_data[img_name] = attrs
            return

        # CSV fallback
        if os.path.exists(self.attr_file_csv):
            import pandas as pd
            df = pd.read_csv(self.attr_file_csv)
            # Identify image column
            candidate_cols = ['image_id', 'image', 'img', 'filename', 'file_name', 'file']
            img_col = next((c for c in candidate_cols if c in df.columns), None)
            if img_col is None:
                # assume first column is image id
                img_col = df.columns[0]
            # Attribute columns are all non-image columns
            attr_cols = [c for c in df.columns if c != img_col]
            self.all_attr_names = list(attr_cols)
            self.attr_data = {}
            for _, row in df.iterrows():
                img_name = str(row[img_col])
                vals = []
                for c in attr_cols:
                    v = row[c]
                    try:
                        iv = int(v)
                    except Exception:
                        # If boolean or string 'True'/'False'
                        if str(v).lower() in ('true', 'yes', '1'):
                            iv = 1
                        elif str(v).lower() in ('false', 'no', '0'):
                            iv = 0
                        else:
                            iv = 0
                    # Map -1/1 to 0/1 if needed
                    if iv in (-1, 1):
                        iv = (iv + 1) // 2
                    iv = 1 if iv > 0 else 0
                    vals.append(iv)
                self.attr_data[img_name] = vals
            return

        # Auto-discover: try scanning root for a plausible attributes CSV/TXT
        import glob
        import pandas as pd
        # Prefer CSVs
        csv_candidates = glob.glob(os.path.join(self.root, "*.csv"))
        txt_candidates = glob.glob(os.path.join(self.root, "*.txt"))
        # Heuristics: attributes file should contain many binary attribute columns including 'Smiling'
        for path in csv_candidates + txt_candidates:
            try:
                if path.endswith(".csv"):
                    df = pd.read_csv(path)
                    candidate_cols = ['image_id', 'image', 'img', 'filename', 'file_name', 'file']
                    img_col = next((c for c in candidate_cols if c in df.columns), None) or df.columns[0]
                    attr_cols = [c for c in df.columns if c != img_col]
                    if 'Smiling' in attr_cols or 'smiling' in [c.lower() for c in attr_cols]:
                        self.all_attr_names = list(attr_cols)
                        self.attr_data = {}
                        for _, row in df.iterrows():
                            img_name = str(row[img_col])
                            vals = []
                            for c in attr_cols:
                                v = row[c]
                                try:
                                    iv = int(v)
                                except Exception:
                                    if str(v).lower() in ('true', 'yes', '1'):
                                        iv = 1
                                    elif str(v).lower() in ('false', 'no', '0'):
                                        iv = 0
                                    else:
                                        iv = 0
                                if iv in (-1, 1):
                                    iv = (iv + 1) // 2
                                iv = 1 if iv > 0 else 0
                                vals.append(iv)
                            self.attr_data[img_name] = vals
                        return
                else:
                    # TXT: check if celebA-style with header of attributes on second line
                    with open(path, 'r') as f:
                        lines = f.readlines()
                    if len(lines) >= 2 and len(lines[1].strip().split()) >= 10:
                        # Likely an attributes file
                        self.all_attr_names = lines[1].strip().split()
                        self.attr_data = {}
                        for line in lines[2:]:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            img_name = parts[0]
                            attrs = [int(x) for x in parts[1:]]
                            attrs = [(x + 1) // 2 for x in attrs]
                            self.attr_data[img_name] = attrs
                        return
            except Exception:
                continue

        raise FileNotFoundError(
            "Could not find CelebA attributes file. Looked for standard TXT/CSV names and attempted auto-discovery. "
            f"Please ensure attributes CSV/TXT is in {self.root} and includes columns like 'Smiling', 'Young', 'Male'."
        )

    def _load_split(self):
        split_map = {'train': 0, 'val': 1, 'test': 2}
        target_split = split_map[self.split]

        split_data = {}
        # TXT format (official)
        if os.path.exists(self.split_file_txt):
            with open(self.split_file_txt, 'r') as f:
                for line in f:
                    img_name, split_id = line.strip().split()
                    split_data[img_name] = int(split_id)
        # CSV fallback
        elif os.path.exists(self.split_file_csv):
            import pandas as pd
            df = pd.read_csv(self.split_file_csv)
            # Identify columns
            candidate_img = ['image_id', 'image', 'img', 'filename', 'file_name', 'file']
            img_col = next((c for c in candidate_img if c in df.columns), None)
            if img_col is None:
                img_col = df.columns[0]
            candidate_split = ['partition', 'split', 'eval_partition', 'split_id']
            split_col = next((c for c in candidate_split if c in df.columns), None)
            if split_col is None:
                # try second column
                split_col = df.columns[1] if len(df.columns) > 1 else None
            if split_col is None:
                raise ValueError("Could not detect split/partition column in CSV.")
            for _, row in df.iterrows():
                img_name = str(row[img_col])
                val = row[split_col]
                # Normalize split to 0/1/2
                if isinstance(val, str):
                    s = val.strip().lower()
                    if s in ('train', 'training', '0'):
                        sid = 0
                    elif s in ('val', 'valid', 'validation', '1'):
                        sid = 1
                    elif s in ('test', 'testing', '2'):
                        sid = 2
                    else:
                        # default to train
                        sid = 0
                else:
                    try:
                        sid = int(val)
                    except Exception:
                        sid = 0
                split_data[img_name] = sid
        else:
            # Auto-discover: scan root for a CSV with a recognizable split column
            import glob
            import pandas as pd
            found = False
            for path in glob.glob(os.path.join(self.root, "*.csv")):
                try:
                    df = pd.read_csv(path)
                    candidate_img = ['image_id', 'image', 'img', 'filename', 'file_name', 'file']
                    img_col = next((c for c in candidate_img if c in df.columns), None) or df.columns[0]
                    candidate_split = ['partition', 'split', 'eval_partition', 'split_id', 'subset', 'phase']
                    split_col = next((c for c in candidate_split if c in df.columns), None)
                    if split_col is None:
                        continue
                    for _, row in df.iterrows():
                        img_name = str(row[img_col])
                        val = row[split_col]
                        if isinstance(val, str):
                            s = val.strip().lower()
                            if s in ('train', 'training', '0'):
                                sid = 0
                            elif s in ('val', 'valid', 'validation', '1'):
                                sid = 1
                            elif s in ('test', 'testing', '2'):
                                sid = 2
                            else:
                                sid = 0
                        else:
                            try:
                                sid = int(val)
                            except Exception:
                                sid = 0
                        split_data[img_name] = sid
                    found = True
                    break
                except Exception:
                    continue
            if not found:
                # If still nothing, consider all images as belonging to requested split (no filtering)
                split_data = {}

        protected_indices = [self.all_attr_names.index(attr)
                             for attr in self.protected_attributes
                             if attr in self.all_attr_names]
        target_idx = self.all_attr_names.index(self.target_attribute)

        self.image_paths = []
        labels_list = []
        sensitive_list = []

        for img_name, attrs in self.attr_data.items():
            if img_name in split_data and split_data[img_name] != target_split:
                continue
            img_path = os.path.join(self.img_dir, img_name)
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                labels_list.append(attrs[target_idx])
                if len(protected_indices) == 1:
                    sensitive_list.append(attrs[protected_indices[0]])
                elif len(protected_indices) == 2:
                    sensitive_val = attrs[protected_indices[0]] * 2 + attrs[protected_indices[1]]
                    sensitive_list.append(sensitive_val)
                else:
                    sensitive_list.append(0)

        self.labels = np.array(labels_list)
        self.sensitive_features = np.array(sensitive_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = int(self.labels[idx])
        sensitive = int(self.sensitive_features[idx])
        return image, torch.tensor(label, dtype=torch.long), torch.tensor(sensitive, dtype=torch.long)

    def get_calibration_subset(self, num_samples: int = 1000, balanced: bool = True) -> Subset:
        if not balanced or len(self.sensitive_features) == 0:
            indices = np.random.choice(len(self), min(num_samples, len(self)), replace=False)
            return Subset(self, indices.tolist())
        unique_groups = np.unique(self.sensitive_features)
        samples_per_group = max(1, num_samples // len(unique_groups))
        selected_indices: List[int] = []
        for group in unique_groups:
            group_indices = np.where(self.sensitive_features == group)[0]
            n_to_sample = min(samples_per_group, len(group_indices))
            if n_to_sample > 0:
                sampled = np.random.choice(group_indices, n_to_sample, replace=False)
                selected_indices.extend(sampled.tolist())
        if len(selected_indices) == 0:
            selected_indices = list(range(min(num_samples, len(self))))
        return Subset(self, selected_indices)


def create_celeba_dataloaders(
    config: dict,
    batch_size: int = 32,
    num_workers: int = 4,
    protected_attributes: Optional[List[str]] = None,
    target_attribute: str = 'Smiling',
    subsample: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    celeba_cfg = (config or {}).get('data', {}).get('celeba', {})
    root = celeba_cfg.get('root') or celeba_cfg.get('path') or os.path.join('data', 'celeba')
    prot = protected_attributes or celeba_cfg.get('protected_attributes') or ['Male', 'Young']
    target_attr = target_attribute or celeba_cfg.get('target_attribute') or 'Smiling'

    train_ds = CelebADataset(root=root, split='train', protected_attributes=prot,
                             target_attribute=target_attr, subsample=subsample)
    val_ds = CelebADataset(root=root, split='val', protected_attributes=prot,
                           target_attribute=target_attr, subsample=subsample)
    test_ds = CelebADataset(root=root, split='test', protected_attributes=prot,
                            target_attribute=target_attr, subsample=subsample)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


