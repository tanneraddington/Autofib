import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def apply_circular_roi(images, center, radius):
    """
    Apply a circular ROI mask to a batch of images.
    
    Args:
        images: Tensor of shape [B, C, H, W]
        center: (cx, cy) center of the circle in pixel coordinates (x = width, y = height)
        radius: radius of the circle in pixels

    Returns:
        masked_images: Tensor [B, C, H, W] with pixels outside the circle set to 0.
    """
    assert images.dim() == 4, "images should be [B, C, H, W]"
    B, C, H, W = images.shape
    device = images.device
    cx, cy = center

    # Make coordinate grid
    ys = torch.arange(H, device=device).view(H, 1).expand(H, W)
    xs = torch.arange(W, device=device).view(1, W).expand(H, W)

    # Compute squared distance to center
    dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
    mask = (dist2 <= radius ** 2).float()  # [H, W]

    # Add batch + channel dims so it broadcasts over [B, C, H, W]
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # Broadcast and apply
    masked_images = images * mask  # [B, C, H, W]
    return masked_images


class ImageBackboneEncoder(nn.Module):
    """
    Wraps a torchvision backbone (ResNet50 or ViT-B/16) and outputs a feature vector per image.
    """
    def __init__(self, backbone='resnet50', train_backbone=False):
        super().__init__()

        self.backbone_name = backbone

        if backbone == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Remove the final FC layer, keep everything up to the global pooling
            self.backbone = nn.Sequential(*list(base_model.children())[:-2])  # [B, 2048, H', W']
            self.feature_dim = 2048
        
        elif backbone == "eff":
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.backbone = base.features  # (B,1280,?,?)
            self.feature_dim = 1280

        elif backbone == 'vit':
            base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            # Remove classification head, keep feature extractor
            if hasattr(base_model, 'heads'):
                base_model.heads = nn.Identity()
            self.backbone = base_model
            self.feature_dim = 768



        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Optionally freeze backbone parameters
        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x: [B, C, H, W], already normalized and resized for the chosen backbone.
        Returns: features [B, D]
        """
        if self.backbone_name == 'resnet50':
            # Convolutional feature map
            with torch.no_grad():
                feats = self.backbone(x)  # [B, 2048, h, w]
            # Global average pooling
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            feats = feats.view(feats.size(0), -1)  # [B, 2048]
        
        elif self.backbone_name == 'eff':
            with torch.no_grad():
                feats = self.backbone(x)  # [B, 1280, h, w]
            # Global average pooling
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            feats = feats.view(feats.size(0), -1)  # [B, 1280]

        elif self.backbone_name == 'vit':
            # ViT expects [B, C, H, W] and outputs [B, D] since we set heads = Identity
            with torch.no_grad():
                feats = self.backbone(x)  # [B, 768]

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        return feats


class ROIFeatureDistance(nn.Module):
    """
    Computes a feature distance between pre and post images within a circular ROI.

    Usage:
        roi_dist = ROIFeatureDistance(backbone='resnet50', distance='cosine')
        d = roi_dist(img_pre, img_post, center=(cx, cy), radius=r)  # [B]
    """
    def __init__(self,
                 backbone='resnet50',
                 distance='cosine',
                 train_backbone=False):
        super().__init__()

        assert distance in ['cosine', 'l2'], "distance must be 'cosine' or 'l2'"
        self.distance_type = distance

        self.encoder = ImageBackboneEncoder(backbone=backbone,
                                            train_backbone=train_backbone)

    def forward(self, img_pre, img_post, center, radius):
        """
        img_pre, img_post: [B, C, H, W]
        center: (cx, cy)
        radius: float
        
        Returns:
            dist: [B] scalar distance per pair
        """
        # Apply the same circular ROI to both
        img_pre_roi = apply_circular_roi(img_pre, center, radius)
        img_post_roi = apply_circular_roi(img_post, center, radius)

        # Encode to feature vectors
        feats_pre = self.encoder(img_pre_roi)   # [B, D]
        feats_post = self.encoder(img_post_roi) # [B, D]

        if self.distance_type == 'cosine':
            # distance = 1 - cosine similarity
            dist = 1.0 - F.cosine_similarity(feats_pre, feats_post, dim=-1)
        else:  # 'l2'
            dist = torch.sqrt(((feats_pre - feats_post) ** 2).sum(dim=-1) + 1e-8)

        return dist  # [B]

