import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class AIvsRealDataset(Dataset):
    """Dataset class for AI-generated vs real images classification."""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_paths (list): List of paths to the images
            labels (list): List of labels (0 for real, 1 for AI-generated)
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label) where image is a transformed image and label is 0 or 1
        """
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image if the image cannot be loaded
            image = Image.new('RGB', (224, 224), color='black')
            
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def load_data(data_dir, batch_size=32, test_size=0.2, val_size=0.1, seed=42):
    """
    Load and prepare datasets for training, validation, and testing.
    
    Args:
        data_dir (str): Path to the dataset directory containing 'real' and 'fake' subdirectories
        batch_size (int): Batch size for dataloaders
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names, class_weights)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Define the transformations for training data (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define the transformations for validation and test data (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Class names
    class_names = ['Real', 'AI-generated']
    
    # Load all image paths and labels
    image_paths = []
    labels = []
    
    # Load real images (label 0)
    real_dir = os.path.join(data_dir, 'Real')
    if not os.path.exists(real_dir):
        raise FileNotFoundError(f"Directory not found: {real_dir}")
    
    for img_name in os.listdir(real_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            img_path = os.path.join(real_dir, img_name)
            image_paths.append(img_path)
            labels.append(0)  # Real images are labeled as 0
    
    # Load fake/AI images (label 1)
    fake_dir = os.path.join(data_dir, 'Fake')
    if not os.path.exists(fake_dir):
        raise FileNotFoundError(f"Directory not found: {fake_dir}")
    
    for img_name in os.listdir(fake_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            img_path = os.path.join(fake_dir, img_name)
            image_paths.append(img_path)
            labels.append(1)  # AI-generated/fake images are labeled as 1
    
    # Calculate class weights for imbalanced dataset
    class_counts = [labels.count(0), labels.count(1)]
    total_samples = len(labels)
    class_weights = torch.FloatTensor([total_samples / (2 * count) if count > 0 else 1.0 for count in class_counts])
    
    print(f"Dataset statistics:")
    print(f"Total images: {total_samples}")
    print(f"Real images: {class_counts[0]} ({class_counts[0]/total_samples*100:.2f}%)")
    print(f"AI-generated images: {class_counts[1]} ({class_counts[1]/total_samples*100:.2f}%)")
    
    # Split data into train+val and test sets
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    
    # Split train+val into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_size/(1-test_size), 
        random_state=seed, stratify=train_val_labels
    )
    
    print(f"Training set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    print(f"Test set: {len(test_paths)} images")
    
    # Create datasets
    train_dataset = AIvsRealDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = AIvsRealDataset(val_paths, val_labels, transform=val_test_transform)
    test_dataset = AIvsRealDataset(test_paths, test_labels, transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_names, class_weights


def visualize_data_distribution(data_dir):
    """
    Visualize the distribution of real and AI-generated images in the dataset.
    
    Args:
        data_dir (str): Path to the dataset directory
    """
    real_dir = os.path.join(data_dir, 'Real')
    fake_dir = os.path.join(data_dir, 'Fake')
    
    real_count = len([f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))])
    fake_count = len([f for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=['Real Images', 'AI-generated Images'], y=[real_count, fake_count])
    plt.title('Dataset Distribution')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('data_distribution.png')
    plt.close()
    
    print(f"Data distribution visualization saved as 'data_distribution.png'")


def visualize_sample_images(data_dir, num_samples=5):
    """
    Visualize sample images from both real and AI-generated categories.
    
    Args:
        data_dir (str): Path to the dataset directory
        num_samples (int): Number of samples to visualize from each category
    """
    real_dir = os.path.join(data_dir, 'Real')
    fake_dir = os.path.join(data_dir, 'Fake')
    
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    
    # Randomly sample images
    if len(real_images) >= num_samples and len(fake_images) >= num_samples:
        real_samples = random.sample(real_images, num_samples)
        fake_samples = random.sample(fake_images, num_samples)
        
        # Create a figure
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        
        # Display real images
        for i, img_path in enumerate(real_samples):
            img = Image.open(img_path).convert('RGB')
            axes[0, i].imshow(img)
            axes[0, i].set_title('Real')
            axes[0, i].axis('off')
        
        # Display AI-generated images
        for i, img_path in enumerate(fake_samples):
            img = Image.open(img_path).convert('RGB')
            axes[1, i].imshow(img)
            axes[1, i].set_title('AI-generated')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images.png')
        plt.close()
        
        print(f"Sample images visualization saved as 'sample_images.png'")
    else:
        print("Not enough images to visualize samples")


if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/your/dataset"  # Replace with the actual path
    
    # Visualize data distribution
    try:
        visualize_data_distribution(data_dir)
        visualize_sample_images(data_dir)
    except Exception as e:
        print(f"Visualization error: {e}")
    
    # Load data
    try:
        train_loader, val_loader, test_loader, class_names, class_weights = load_data(data_dir)
        print("Data loaders created successfully")
    except Exception as e:
        print(f"Error loading data: {e}")
