"""
Dataset handlers for CelebA, UTKFace, and FairFace
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import warnings
from sklearn.model_selection import train_test_split

class FairnessDataset(Dataset):
    """Base class for fairness-aware datasets"""
    
    def __init__(self, 
                 root: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 protected_attributes: List[str] = None,
                 target_attribute: str = None):
        """
        Args:
            root: Root directory of dataset
            split: 'train', 'val', or 'test'
            transform: Image transformations
            protected_attributes: List of protected attribute names
            target_attribute: Target attribute for prediction
        """
        self.root = root
        self.split = split
        self.transform = transform or self._default_transform()
        self.protected_attributes = protected_attributes or []
        self.target_attribute = target_attribute
        
        # To be implemented by subclasses
        self.image_paths = []
        self.labels = []
        self.sensitive_features = []
        
    def _default_transform(self):
        """Default image transformation"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get item by index"""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        sensitive = self.sensitive_features[idx]
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'sensitive': torch.tensor(sensitive, dtype=torch.long),
            'idx': idx
        }
    
    def get_group_statistics(self) -> Dict[str, Any]:
        """Get statistics about demographic groups"""
        unique_groups = np.unique(self.sensitive_features)
        stats = {}
        
        for group in unique_groups:
            mask = self.sensitive_features == group
            group_size = mask.sum()
            
            # Label distribution within group
            group_labels = self.labels[mask]
            unique_labels, counts = np.unique(group_labels, return_counts=True)
            
            stats[f'group_{group}'] = {
                'size': int(group_size),
                'proportion': float(group_size / len(self)),
                'label_distribution': {
                    int(label): int(count) 
                    for label, count in zip(unique_labels, counts)
                }
            }
        
        return stats


class CelebADataset(FairnessDataset):
    """CelebA dataset handler with fairness attributes"""
    
    def __init__(self, 
                 root: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 protected_attributes: List[str] = ['Male', 'Young'],
                 target_attribute: str = 'Smiling',
                 subsample: Optional[int] = None):
        """
        CelebA dataset with demographic attributes
        
        Args:
            root: Root directory containing CelebA
            split: 'train', 'val', or 'test'
            transform: Image transformations
            protected_attributes: Protected demographic attributes
            target_attribute: Target prediction attribute
            subsample: Optional subsampling for faster experimentation
        """
        super().__init__(root, split, transform, protected_attributes, target_attribute)
        
        self.img_dir = os.path.join(root, 'img_align_celeba')
        self.attr_file = os.path.join(root, 'list_attr_celeba.txt')
        self.split_file = os.path.join(root, 'list_eval_partition.txt')
        
        # Load attributes
        self._load_attributes()
        self._load_split()
        
        if subsample and subsample < len(self.image_paths):
            indices = np.random.choice(len(self.image_paths), subsample, replace=False)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = self.labels[indices]
            self.sensitive_features = self.sensitive_features[indices]
    
    def _load_attributes(self):
        """Load CelebA attributes"""
        # Read attribute file
        with open(self.attr_file, 'r') as f:
            lines = f.readlines()
        
        # Parse header (attribute names)
        self.all_attr_names = lines[1].strip().split()
        
        # Parse attributes for each image
        self.attr_data = {}
        for line in lines[2:]:
            parts = line.strip().split()
            img_name = parts[0]
            attrs = [int(x) for x in parts[1:]]
            # Convert -1,1 to 0,1
            attrs = [(x + 1) // 2 for x in attrs]
            self.attr_data[img_name] = attrs
    
    def _load_split(self):
        """Load train/val/test split"""
        split_map = {'train': 0, 'val': 1, 'test': 2}
        target_split = split_map[self.split]
        
        # Read split file
        split_data = {}
        if os.path.exists(self.split_file):
            with open(self.split_file, 'r') as f:
                for line in f:
                    img_name, split_id = line.strip().split()
                    split_data[img_name] = int(split_id)
        
        # Get indices for attributes
        if self.protected_attributes:
            protected_indices = [self.all_attr_names.index(attr) 
                               for attr in self.protected_attributes 
                               if attr in self.all_attr_names]
        else:
            protected_indices = []
        
        target_idx = self.all_attr_names.index(self.target_attribute)
        
        # Build dataset
        self.image_paths = []
        labels_list = []
        sensitive_list = []
        
        for img_name, attrs in self.attr_data.items():
            # Check split
            if img_name in split_data and split_data[img_name] != target_split:
                continue
            
            img_path = os.path.join(self.img_dir, img_name)
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                labels_list.append(attrs[target_idx])
                
                # Combine protected attributes into single value
                if len(protected_indices) == 1:
                    sensitive_list.append(attrs[protected_indices[0]])
                elif len(protected_indices) == 2:
                    # Create 4 groups: 00, 01, 10, 11
                    sensitive_val = attrs[protected_indices[0]] * 2 + attrs[protected_indices[1]]
                    sensitive_list.append(sensitive_val)
                else:
                    sensitive_list.append(0)  # Default group
        
        self.labels = np.array(labels_list)
        self.sensitive_features = np.array(sensitive_list)


class UTKFaceDataset(FairnessDataset):
    """UTKFace dataset handler with age, gender, race attributes"""
    
    def __init__(self,
                 root: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 protected_attributes: List[str] = ['gender', 'race'],
                 target_attribute: str = 'age_group',
                 age_groups: List[Tuple[int, int]] = [(0, 18), (19, 35), (36, 60), (61, 120)],
                 subsample: Optional[int] = None):
        """
        UTKFace dataset
        
        Args:
            root: Root directory containing UTKFace
            split: 'train', 'val', or 'test'
            transform: Image transformations
            protected_attributes: Protected demographic attributes
            target_attribute: Target prediction attribute
            age_groups: Age group boundaries for classification
            subsample: Optional subsampling
        """
        super().__init__(root, split, transform, protected_attributes, target_attribute)
        
        self.age_groups = age_groups
        self._load_data()
        
        if subsample and subsample < len(self.image_paths):
            indices = np.random.choice(len(self.image_paths), subsample, replace=False)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = self.labels[indices]
            self.sensitive_features = self.sensitive_features[indices]
    
    def _load_data(self):
        """Load UTKFace data"""
        # UTKFace filenames format: [age]_[gender]_[race]_[date&time].jpg
        self.image_paths = []
        ages = []
        genders = []
        races = []
        
        for filename in os.listdir(self.root):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                try:
                    parts = filename.split('_')
                    if len(parts) >= 4:
                        age = int(parts[0])
                        gender = int(parts[1])  # 0: male, 1: female
                        race = int(parts[2])     # 0-4: different races
                        
                        self.image_paths.append(os.path.join(self.root, filename))
                        ages.append(age)
                        genders.append(gender)
                        races.append(race)
                except:
                    continue
        
        # Convert to numpy arrays
        ages = np.array(ages)
        genders = np.array(genders)
        races = np.array(races)
        
        # Create age groups
        age_group_labels = np.zeros(len(ages))
        for i, (min_age, max_age) in enumerate(self.age_groups):
            mask = (ages >= min_age) & (ages <= max_age)
            age_group_labels[mask] = i
        
        # Set labels based on target attribute
        if self.target_attribute == 'age_group':
            self.labels = age_group_labels.astype(int)
        elif self.target_attribute == 'gender':
            self.labels = genders
        elif self.target_attribute == 'race':
            self.labels = races
        else:
            self.labels = age_group_labels.astype(int)
        
        # Set sensitive features
        if 'gender' in self.protected_attributes and 'race' in self.protected_attributes:
            # Create combined groups
            self.sensitive_features = genders * 5 + races  # 10 total groups
        elif 'gender' in self.protected_attributes:
            self.sensitive_features = genders
        elif 'race' in self.protected_attributes:
            self.sensitive_features = races
        else:
            self.sensitive_features = np.zeros(len(ages))
        
        # Split data
        self._create_splits()
    
    def _create_splits(self):
        """Create train/val/test splits"""
        n_samples = len(self.image_paths)
        indices = np.arange(n_samples)
        
        # Stratified split based on sensitive features
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=self.sensitive_features, random_state=42
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.25, stratify=self.sensitive_features[train_idx], random_state=42
        )
        
        # Apply split
        if self.split == 'train':
            selected_idx = train_idx
        elif self.split == 'val':
            selected_idx = val_idx
        else:
            selected_idx = test_idx
        
        self.image_paths = [self.image_paths[i] for i in selected_idx]
        self.labels = self.labels[selected_idx]
        self.sensitive_features = self.sensitive_features[selected_idx]


class FairFaceDataset(FairnessDataset):
    """FairFace dataset handler"""
    
    def __init__(self,
                 root: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 protected_attributes: List[str] = ['gender', 'race'],
                 target_attribute: str = 'age',
                 subsample: Optional[int] = None):
        """
        FairFace dataset
        
        Args:
            root: Root directory containing FairFace
            split: 'train' or 'val'
            transform: Image transformations
            protected_attributes: Protected demographic attributes
            target_attribute: Target prediction attribute
            subsample: Optional subsampling
        """
        super().__init__(root, split, transform, protected_attributes, target_attribute)
        
        self.labels_file = os.path.join(root, f'fairface_label_{split}.csv')
        self.img_dir = os.path.join(root, split)
        
        self._load_data()
        
        if subsample and subsample < len(self.image_paths):
            indices = np.random.choice(len(self.image_paths), subsample, replace=False)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = self.labels[indices]
            self.sensitive_features = self.sensitive_features[indices]
    
    def _load_data(self):
        """Load FairFace data from CSV"""
        # Read CSV file
        df = pd.read_csv(self.labels_file)
        
        # Map categorical values to integers
        age_map = {'0-2': 0, '3-9': 1, '10-19': 2, '20-29': 3, 
                   '30-39': 4, '40-49': 5, '50-59': 6, '60-69': 7, 'more than 70': 8}
        gender_map = {'Male': 0, 'Female': 1}
        race_map = {'White': 0, 'Black': 1, 'Latino_Hispanic': 2, 
                    'East Asian': 3, 'Southeast Asian': 4, 'Indian': 5, 'Middle Eastern': 6}
        
        # Process data
        self.image_paths = []
        labels_list = []
        sensitive_list = []
        
        for _, row in df.iterrows():
            img_path = os.path.join(self.img_dir, row['file'])
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                
                # Get label based on target attribute
                if self.target_attribute == 'age':
                    label = age_map.get(row['age'], 0)
                elif self.target_attribute == 'gender':
                    label = gender_map.get(row['gender'], 0)
                elif self.target_attribute == 'race':
                    label = race_map.get(row['race'], 0)
                else:
                    label = age_map.get(row['age'], 0)
                
                labels_list.append(label)
                
                # Get sensitive features
                gender_val = gender_map.get(row['gender'], 0)
                race_val = race_map.get(row['race'], 0)
                
                if 'gender' in self.protected_attributes and 'race' in self.protected_attributes:
                    sensitive_val = gender_val * 7 + race_val  # 14 total groups
                elif 'gender' in self.protected_attributes:
                    sensitive_val = gender_val
                elif 'race' in self.protected_attributes:
                    sensitive_val = race_val
                else:
                    sensitive_val = 0
                
                sensitive_list.append(sensitive_val)
        
        self.labels = np.array(labels_list)
        self.sensitive_features = np.array(sensitive_list)


def get_fairness_dataloader(dataset_name: str,
                           root: str,
                           split: str = 'train',
                           batch_size: int = 32,
                           num_workers: int = 4,
                           shuffle: bool = True,
                           protected_attributes: List[str] = None,
                           target_attribute: str = None,
                           subsample: Optional[int] = None) -> DataLoader:
    """
    Get dataloader for a fairness dataset
    
    Args:
        dataset_name: 'celeba', 'utkface', or 'fairface'
        root: Root directory of dataset
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        protected_attributes: Protected demographic attributes
        target_attribute: Target attribute for prediction
        subsample: Optional subsampling for faster experiments
        
    Returns:
        DataLoader instance
    """
    if dataset_name.lower() == 'celeba':
        dataset = CelebADataset(
            root=root,
            split=split,
            protected_attributes=protected_attributes or ['Male', 'Young'],
            target_attribute=target_attribute or 'Smiling',
            subsample=subsample
        )
    elif dataset_name.lower() == 'utkface':
        dataset = UTKFaceDataset(
            root=root,
            split=split,
            protected_attributes=protected_attributes or ['gender', 'race'],
            target_attribute=target_attribute or 'age_group',
            subsample=subsample
        )
    elif dataset_name.lower() == 'fairface':
        dataset = FairFaceDataset(
            root=root,
            split=split,
            protected_attributes=protected_attributes or ['gender', 'race'],
            target_attribute=target_attribute or 'age',
            subsample=subsample
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader


def create_balanced_calibration_set(dataset: FairnessDataset,
                                   n_samples: int = 1000) -> Subset:
    """
    Create a balanced calibration set for quantization
    
    Args:
        dataset: Fairness dataset
        n_samples: Number of samples to include
        
    Returns:
        Subset with balanced demographic representation
    """
    # Get unique groups
    unique_groups = np.unique(dataset.sensitive_features)
    samples_per_group = n_samples // len(unique_groups)
    
    selected_indices = []
    for group in unique_groups:
        group_indices = np.where(dataset.sensitive_features == group)[0]
        
        # Sample from group
        n_to_sample = min(samples_per_group, len(group_indices))
        sampled = np.random.choice(group_indices, n_to_sample, replace=False)
        selected_indices.extend(sampled)
    
    return Subset(dataset, selected_indices)
