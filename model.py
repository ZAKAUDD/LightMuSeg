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
        
        #self.trans = AIM(iC_list=(16, 24, 32, 96, 320), oC_list=(16, 24, 32, 96, 320))
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
