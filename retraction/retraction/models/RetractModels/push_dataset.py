import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pickle

class PushDataset(Dataset):
    def __init__(self, data_path, transform=None, max_samples=None):
        """
        Args:
            data_path (str): Path to the directory containing pickle files
            transform (callable, optional): Optional transform to be applied on images
            max_samples (int, optional): Maximum number of samples to load
        """
        self.transform = transform
        self.data_samples = []
        
        # Check if data_path is a directory
        if not os.path.isdir(data_path):
            raise ValueError(f"data_path must be a directory: {data_path}")
        
        # Get all pickle files in the directory
        pickle_files = [f for f in os.listdir(data_path) if f.endswith('.pkl') or f.endswith('.pickle')]
        pickle_files.sort()  # Sort for consistency
        
        # Limit the number of files if max_samples is specified
        if max_samples is not None and max_samples < len(pickle_files):
            pickle_files = pickle_files[:max_samples]
        
        # Load each pickle file
        for filename in pickle_files:
            file_path = os.path.join(data_path, filename)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    self.data_samples.append({
                        'image': data['image'],
                        'start_pose': data['start_pose'],
                        'end_pose': data['end_pose'],
                        'displacement': data['displacement'],
                    })
                    # print(f"Loaded {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        image = sample['image']
        start_pose = torch.tensor(sample['start_pose'], dtype=torch.float32)
        # TODO: verify this is better than end_pose
        displacement = torch.tensor(sample['displacement'], dtype=torch.float32)

        # Convert image to PIL if it's a numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, start_pose, displacement

def create_push_data_loaders(data_path, batch_size=16, transform=None, max_samples=None, 
                            val_split=0.2, shuffle=True, num_workers=4, seed=42):
    """
    Create train and validation data loaders for push dataset
    
    Args:
        data_path (str): Path to the directory containing pickle files
        batch_size (int): Batch size for the data loaders
        transform (callable, optional): Transform to apply to the images
        max_samples (int, optional): Maximum number of samples to load
        val_split (float): Fraction of data to use for validation
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        seed (int): Random seed for reproducibility
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create the dataset
    dataset = PushDataset(data_path, transform=transform, max_samples=max_samples)
    
    # Print dataset size
    print(f"Total dataset size: {len(dataset)}")
    
    # Split into train and validation sets
    if val_split > 0:
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        return train_loader, val_loader
    else:
        # No validation split, return only train loader
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        return train_loader, None