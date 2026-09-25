import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, random_split
import os
import torchvision.models as models

class Push2dCVAE(nn.Module):
    def __init__(self, 
                 n_latent_dims=7,
                 n_hidden_units=64,
                 image_feature_dim=8,
                 backbone='resnet50',
                 device="cpu"
                 ):
        super().__init__()

        # TODO: add a smaall MLP for the actions maybe? as a little pre encoder?

        self.device = device
        self.action_dim = 6

        if backbone == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone_encoder = nn.Sequential(*list(base_model.children())[:-2])
            feature_channels = 2048
        
        elif backbone == 'vit':
            base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            self.backbone_encoder= base_model
            feature_channels = 768
        
        elif backbone == "eff":
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.backbone_encoder= base.features  # (B,1280,?,?)
            feature_channels = 1280

        
        # Freeze backbone parameters for transfer learning
        for param in self.backbone_encoder.parameters():
            param.requires_grad = False

        self.image_feature_encoder = nn.Sequential(
            nn.Linear(feature_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64,image_feature_dim )
        )

        self.encoder = nn.Sequential(
            nn.Linear(self.action_dim+image_feature_dim, n_hidden_units), # 14 to 64
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units//2), # 64 to 32
            nn.ReLU()
        )


        # 7 dimensional latent space
        self.mu = nn.Linear(n_hidden_units//2, n_latent_dims)
        self.logvar = nn.Linear(n_hidden_units//2, n_latent_dims)

        self.decoder = nn.Sequential(
            nn.Linear(n_latent_dims+image_feature_dim, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, self.action_dim),  # 6 values: 3 for start action, 3 for end action
        )
        
        # Split the decoder output into start and end actions

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, image, action):
        """
        image: input image
        c: conditioning variables
        """
        # Process image through backbone
        with torch.no_grad():
            features = self.backbone_encoder(image)
        
        # Reshape features for the linear layer
        # ResNet50 outputs features with shape [batch_size, 2048, h, w]
        # We need to convert this to [batch_size, 2048]
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        
        # Encode image features to lower dimension
        image_features = self.image_feature_encoder(features)
        
        # Combine image features with conditioning variables
        x_combined = torch.cat([action, image_features], dim=1)
        
        # Run the encoder to generate latent space parameters
        h = self.encoder(x_combined)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # Concatenate latent space with conditioning variables (the image features)
        z_combined = torch.cat([z, image_features], dim=1)
        decoded = self.decoder(z_combined)
        
        # Split into start and end actions (each with 3 components)
        start_action = decoded[:, :3]
        end_action = decoded[:, 3:]
        
        return start_action, end_action, mu, logvar
    
    def sample(self, image, n_samples=1, device="cpu"):
        """
        Sample actions conditioned on image features
        image: input image
        n_samples: number of samples to generate
        """
        actions = []
        with torch.no_grad():
            # Process image through backbone
            features = self.backbone_encoder(image)
            
            # Apply the same pooling as in forward
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
            
            # Encode image features to lower dimension
            image_features = self.image_feature_encoder(features)
            
            # For sampling, we need to generate multiple samples from the prior distribution
            # In a CVAE, we sample from the standard normal distribution N(0,1)
            # This is the prior distribution that the encoder is trained to match
            for i in range(n_samples):
                # Sample from the prior distribution
                z = torch.randn(image_features.size(0), self.mu.out_features).to(image_features.device)
                
                # Combine with image features
                z_combined = torch.cat([z, image_features], dim=1)
                
                # Decode to get actions
                action = self.decoder(z_combined)
                
                # Split into start and end actions
                # Squeeze to remove the batch dimension, ensuring shape is (3,) not (1, 3)
                start_action = action[:, :3].squeeze(0)
                end_action = action[:, 3:].squeeze(0)
                
                # Store as tensors (not numpy arrays)
                actions.append((start_action, end_action))
                print(f"Sampled action {i}: start={start_action}, end={end_action}")
                
        return actions
    
    def loss(self, x, start_action, end_action, mu, logvar):
        """
        x: ground truth actions (combined start and end)
        start_action: predicted start action (x, y, z)
        end_action: predicted end action (x, y, z)
        mu, logvar: latent distribution parameters
        """
        # Split ground truth into start and end actions
        start_gt = x[:, :3]
        end_gt = x[:, 3:]
        beta = 0.001
        
        # MSE for reconstruction loss (better for continuous action values)
        start_recon_loss = F.mse_loss(start_action, start_gt, reduction='sum')
        end_recon_loss = F.mse_loss(end_action, end_gt, reduction='sum')
        recon_loss = start_recon_loss + end_recon_loss
        
        # KL divergence (compared to N(0, 1))
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + kl_div * beta
        total_loss /= x.size(0)  # divide by batch size
        
        return total_loss, recon_loss/x.size(0), kl_div/x.size(0)