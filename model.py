import torch
from torch.nn.modules.activation import PReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
#from modules import*
import torchvision.models as models

#################################modules##############################
from torchvision.models import mobilenet_v2
import torch
from torch.nn.modules.activation import PReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Conv2d, Parameter, Softmax
from typing import Optional
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.dropout = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, attention_heads=4, reduction_ratio=4, dropout=0.1):
        """
        Multi-Scale Feature Fusion module with attention.
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            attention_heads: Number of heads in the attention mechanism.
            reduction_ratio: Reduction ratio for channel attention.
            dropout: Dropout rate for attention layers.
        """
        super(MultiScaleFeatureFusion, self).__init__()

        # Multi-scale convolutions
        self.scale1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.scale2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.scale3 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.reduce_channel = nn.Conv2d(out_channels*3, out_channels, kernel_size=1, padding=0)
        # Attention mechanism
        #self.attention = #nn.MultiheadAttention(embed_dim=out_channels, num_heads=attention_heads, dropout=dropout)
        self.spatialatt = SpatialAttentionModule()
        # Channel attention (squeeze-and-excitation)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(out_channels, out_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(out_channels // reduction_ratio, out_channels, kernel_size=1)

        # Fusion convolution
        self.fusion_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass for multi-scale feature fusion.
        Args:
            x: Input tensor of shape (B, C, H, W).
        Returns:
            Fused feature tensor of shape (B, out_channels, H, W).
        """
        # Multi-scale feature extraction
        scale1_features = self.scale1(x)  # Shape: (B, out_channels, H, W)
        scale2_features = self.scale2(x)  # Shape: (B, out_channels, H, W)
        scale3_features = self.scale3(x)  # Shape: (B, out_channels, H, W)
        #print(scale1_features.shape,scale2_features.shape,scale3_features.shape)
        # Combine features from different scales
        multi_scale_features = self.reduce_channel(torch.cat([scale1_features, scale2_features, scale3_features], dim=1))
        #print('concated',multi_scale_features.shape)
        # Channel attention
        b, c, h, w = multi_scale_features.size()
        squeeze = self.global_pool(multi_scale_features)  # Shape: (B, C, 1, 1)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))  # Shape: (B, C, 1, 1)
        attended_features = multi_scale_features * excitation  # Apply channel attention
        
        # Spatial attention
        # spatial_features = attended_features.view(b, c, -1).permute(2, 0, 1)  # Shape: (H*W, B, C)
        # spatial_features, _ = self.attention(spatial_features, spatial_features, spatial_features)
        # spatial_features = spatial_features.permute(1, 2, 0).view(b, c, h, w)  # Reshape back to (B, C, H, W)
        spatial_features= self.spatialatt(attended_features)
        #refined_features = spatial_features + attended_features
        # Final feature fusion
        fused_features = self.fusion_conv(spatial_features)  # Shape: (B, out_channels, H, W)

        return fused_features


class FeatureRefinementAndFusion(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        """
        Feature Refinement and Fusion Module (FRFM).
        Args:
            in_channels: Number of input channels for the feature maps.
            reduction_ratio: Reduction ratio for channel attention. Defaults to 4.
        """
        super(FeatureRefinementAndFusion, self).__init__()
        
        # Channel Attention
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

        # Spatial Fusion via Lightweight Convolution
        self.spatial_fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x1, x2):
        """
        Forward pass for the module.
        Args:
            x1: First input feature map of shape (B, C, H, W).
            x2: Second input feature map of shape (B, C, H, W).
        Returns:
            Refined and fused feature map of shape (B, C, H, W).
        """
        # Ensure x1 and x2 have the same dimensions
        if x1.shape != x2.shape:
            raise ValueError(f"Input shapes must match, but got {x1.shape} and {x2.shape}")

        # Step 1: Channel Attention
        # Combine features for channel attention
        combined = x1 + x2  # Element-wise addition
        b, c, _, _ = combined.size()
        avg_pooled = self.global_pool(combined).view(b, c)  # Shape: (B, C)
        channel_att = self.channel_fc(avg_pooled).view(b, c, 1, 1)  # Shape: (B, C, 1, 1)
        refined_x1 = x1 * channel_att  # Channel-wise refinement
        refined_x2 = x2 * channel_att  # Channel-wise refinement

        # Step 2: Spatial Fusion
        # Concatenate the refined features along the channel dimension
        spatial_fusion_input = torch.cat([refined_x1, refined_x2], dim=1)  # Shape: (B, 2*C, H, W)
        fused_features = self.spatial_fusion(spatial_fusion_input)  # Shape: (B, C, H, W)

        return fused_features

class NewModelMBLV2(nn.Module):
    '''
        mmobilenet v2 + unet 
    '''
 
    def __init__(self, classes=1):
 
        super(NewModelMBLV2, self).__init__()
        # -----------------------------------------------------------------
        # encoder  
        # ---------------------
         # ---------------------
         #self.feature = mobilenet_v2()
        mobilenet = mobilenet_v2(pretrained=True)
        # Freeze all layers
        # for param in mobilenet.parameters():
        #     param.requires_grad = False

        self.encoder = mobilenet.features
        #s1, s2, s3, s4, s5 = self.feature(input)
        
       
        # -----------------------------------------------------------------
        # decoder 
        # ---------------------
        self.mltslayer1 = MultiScaleFeatureFusion(16,16)
        self.mltslayer2 = MultiScaleFeatureFusion(24,24)
        self.mltslayer3 = MultiScaleFeatureFusion(32,32)
        self.mltslayer4 = MultiScaleFeatureFusion(96,96)
        self.mltslayer5 = MultiScaleFeatureFusion(320,320)
        self.s5_up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        #FPEBlock(320, 320,[1,2,4,8]),
                                        nn.Conv2d(320, 96, 3, 1, 1),
                                        nn.BatchNorm2d(96),
                                        nn.ReLU())
        
        self.sideout4 = SideoutBlock(96, 1)
        self.s4_up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        #FPEBlock(96, 96,[1,2,4,8]),
                                        nn.Conv2d(96, 32, 3, 1, 1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU())
        
        self.sideout3 = SideoutBlock(32, 1)
        self.s3_up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        #FPEBlock(32, 32,[1,2,4,8]),
                                        nn.Conv2d(32, 24, 3, 1, 1),
                                        nn.BatchNorm2d(24),
                                        nn.ReLU())
    
        self.sideout2 = SideoutBlock(24, 1)
        self.s2_up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        #FPEBlock(24,24,[1,2,4,8]),
                                        nn.Conv2d(24, 16, 3, 1, 1),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU())
 
        self.last_conv = nn.Conv2d(16, classes, 3, 1, 1)
        self.last_up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.last_up3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.last_up4 = nn.Upsample(scale_factor=8, mode='bilinear')
        ######outputting the other layers
        self.lca45 =FeatureRefinementAndFusion(96)#FeatureManipulationAndFusion(96)#LCA()
        self.lca34 =FeatureRefinementAndFusion(32)#FeatureManipulationAndFusion(32)#LCA()
        self.lca23 =FeatureRefinementAndFusion(24)#FeatureManipulationAndFusion(24)#LCA()
        self.lca12 =FeatureRefinementAndFusion(16)#FeatureManipulationAndFusion(16)#LCA()
        
    def forward(self, input):
 
        # -----------------------------------------------
        # encoder 
        # ---------------------
        #s1, s2, s3, s4, s5 = self.feature(input)
        s1 = self.encoder[0:2](input) 
        #print(es1.shape)         # Block 0
        s2 = self.encoder[2:4](s1)    # Blocks 1-3, 1:4
        s3 = self.encoder[4:7](s2)    # Blocks 4-6,4:7
        s4 = self.encoder[7:14](s3)   # Blocks 7-13,7:14
        s5 = self.encoder[14:18](s4)  
        #print(s1.shape,s2.shape,s3.shape,s4.shape,s5.shape)
        s1, s2, s3, s4, s5 = self.mltslayer1(s1),self.mltslayer2(s2),self.mltslayer3(s3),self.mltslayer4(s4),self.mltslayer5(s5)#self.trans(s1, s2, s3, s4, s5)
        # -----------------------------------------------
        # decoder
        # ---------------------
        s4_ = self.s5_up_conv(s5)
        s4 = self.lca45(s4, s4_)
        
        #print('s4:',s4.shape)
        s3_ = self.s4_up_conv(s4)
        s3 = self.lca34(s3, s3_)
        
        #print('s3:',s3.shape)
        s2_ = self.s3_up_conv(s3)
        s2 = self.lca23(s2, s2_)
        
        #print('s2:',s2.shape)
        s1_ = self.s2_up_conv(s2)
        s1 = self.lca12(s1, s1_)
        
 
        out = self.last_up(self.last_conv(s1))
        #print(out.shape)
        return out
