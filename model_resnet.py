import torch
import torch.nn as nn
import torchvision.models as models


class ResNetModel(nn.Module):
    """
    ResNet model for AI vs Real image classification.
    
    This model uses a pre-trained ResNet architecture and adds a custom
    classification head for the binary classification task.
    """
    
    def __init__(self, num_classes=2, model_name='resnet50', pretrained=True, freeze_backbone=False):
        """
        Initialize the ResNet model.
        
        Args:
            num_classes (int): Number of output classes (default: 2 for binary classification)
            model_name (str): Which ResNet variant to use ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            pretrained (bool): Whether to use pre-trained weights
            freeze_backbone (bool): Whether to freeze the backbone layers
        """
        super(ResNetModel, self).__init__()
        
        # Select the appropriate ResNet variant
        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 2048
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 2048
        elif model_name == 'resnet152':
            self.backbone = models.resnet152(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Freeze backbone layers if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Remove the original fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add custom classification head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Add attention mechanism (optional)
        self.attention = SpatialAttention()
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Pass through backbone
        features = self.backbone(x)
        
        # Apply classifier
        output = self.classifier(features)
        
        return output
    
    def get_features(self, x):
        """
        Extract features from the backbone network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Feature tensor
        """
        return self.backbone(x)


class SpatialAttention(nn.Module):
    """
    Spatial attention module to focus on important regions of the image.
    """
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Calculate average and max along channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along the channel dimension
        attention = torch.cat([avg_pool, max_pool], dim=1)
        
        # Apply convolution and sigmoid activation
        attention = self.conv(attention)
        attention_mask = self.sigmoid(attention)
        
        # Apply the attention mask
        return x * attention_mask


def initialize_resnet_model(config):
    """
    Initialize and configure the ResNet model based on the provided configuration.
    
    Args:
        config (dict): Configuration parameters for the model
        
    Returns:
        ResNetModel: Configured ResNet model
    """
    model = ResNetModel(
        num_classes=config.get('num_classes', 2),
        model_name=config.get('model_name', 'resnet50'),
        pretrained=config.get('pretrained', True),
        freeze_backbone=config.get('freeze_backbone', False)
    )
    
    # Print model summary
    print(f"Initialized {config.get('model_name', 'resnet50')} model:")
    print(f"Number of classes: {config.get('num_classes', 2)}")
    print(f"Pretrained: {config.get('pretrained', True)}")
    print(f"Freeze backbone: {config.get('freeze_backbone', False)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model


def load_model_checkpoint(model, checkpoint_path):
    """
    Load model weights from a checkpoint.
    
    Args:
        model (nn.Module): The model to load weights into
        checkpoint_path (str): Path to the checkpoint file
        
    Returns:
        nn.Module: Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Loaded model checkpoint from {checkpoint_path}")
    return model


def save_model_checkpoint(model, optimizer, epoch, accuracy, loss, path):
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): The model to save
        optimizer (torch.optim.Optimizer): The optimizer
        epoch (int): Current epoch
        accuracy (float): Validation accuracy
        loss (float): Validation loss
        path (str): Path to save the checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'loss': loss
    }, path)
    
    print(f"Model checkpoint saved to {path}")


if __name__ == "__main__":
    # Example usage
    config = {
        'num_classes': 2,
        'model_name': 'resnet50',
        'pretrained': True,
        'freeze_backbone': False
    }
    
    # Initialize model
    model = initialize_resnet_model(config)
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    outputs = model(x)
    print(f"Output shape: {outputs.shape}")
