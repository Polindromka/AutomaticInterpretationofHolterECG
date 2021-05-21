import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling=True, batch_norm=False):
        super().__init__()
        self.pooling = pooling
        self.batch_norm = batch_norm
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=True, stride=1)
        self.act = nn.ReLU()
        if self.pooling:
            self.maxpool = nn.MaxPool1d(2)
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.act(x)
        if self.pooling:
            x = self.maxpool(x)  
        return x
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, pooling=False)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, pooling=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, pooling=False)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, pooling=False)
            
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
    

    
    
    
class ECGEncoder(nn.Module):    
    def __init__(self, channels, bottleneck=16, batch_norm=True):
        super().__init__()

        self.conv1 = Conv1dBlock(          2, channels[0], 7, pooling=False, batch_norm=batch_norm)
        self.conv2 = Conv1dBlock(channels[0], channels[1], 7, pooling=False, batch_norm=False)
        
        self.down1 = DownBlock(channels[1], channels[2], 5)
        self.down2 = DownBlock(channels[2], channels[3], 5)
        self.down3 = DownBlock(channels[3], channels[4], 5)
        self.down4 = DownBlock(channels[4], channels[5], 3)
        self.down5 = DownBlock(channels[5], channels[6], 3)
        
        self.conv3 = Conv1dBlock(channels[6], channels[6], 3, pooling=False, batch_norm=batch_norm)
        self.conv4 = Conv1dBlock(channels[6], channels[6], 3, pooling=False, batch_norm=False)
        
        self.bottleneck = nn.Conv1d(channels[6], bottleneck, 4)
        self.bottlenect_act = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.bottleneck(x)
        x = self.bottlenect_act(x)
        return x
    
    
class ECGDecoder(nn.Module):
    def __init__(self, channels, bottleneck=16, batch_norm=True):
        super().__init__()
        
        self.bottleneck = nn.Conv1d(bottleneck, channels[6], 1)
        self.bottlenect_act = nn.ReLU()
        
        self.up1 = UpBlock(channels[6], channels[6], 3)
        self.up2 = UpBlock(channels[6], channels[5], 3)
        self.up3 = UpBlock(channels[5], channels[4], 3)
        
        self.conv5 = Conv1dBlock(channels[4], channels[4], 3, pooling=False, batch_norm=batch_norm)
        self.conv6 = Conv1dBlock(channels[4], channels[4], 3, pooling=False, batch_norm=False)        
        
        self.up4 = UpBlock(channels[4], channels[3], 5)
        self.up5 = UpBlock(channels[3], channels[2], 5)
        
        self.conv7 = Conv1dBlock(channels[2], channels[2], 3, pooling=False, batch_norm=batch_norm)
        self.conv8 = Conv1dBlock(channels[2], channels[2], 3, pooling=False, batch_norm=False)        
        
        self.up6 = UpBlock(channels[2], channels[2], 7)
        self.up7 = UpBlock(channels[2], channels[2], 7)
        
        self.conv9 = Conv1dBlock(channels[2], channels[1], 3, pooling=False, batch_norm=batch_norm)
        self.conv10 = Conv1dBlock(channels[1], channels[0], 3, pooling=False, batch_norm=False) 
        
        self.output = nn.Conv1d(channels[0], 2, 1)  
       
    
    def forward(self, x):  
        x = self.bottleneck(x)
        x = self.bottlenect_act(x)
        
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = self.up4(x)
        x = self.up5(x)
        
        x = self.conv7(x)
        x = self.conv8(x)
        
        x = self.up6(x)
        x = self.up7(x)
        
        x = self.conv9(x)
        x = self.conv10(x)
        
        x = self.output(x) 
        return x
        
        
    
class ECGAutoEncoder(nn.Module):
    def __init__(self, channels_mult=1, bottleneck=16, batch_norm=True):
        super().__init__()
        
        channels = (16, 24, 32, 40, 48, 56, 64)
        channels = (np.array(channels) * channels_mult).astype('int32')
        
        self.encoder = ECGEncoder(channels, bottleneck=bottleneck, batch_norm=batch_norm)
        self.decoder = ECGDecoder(channels, bottleneck=bottleneck, batch_norm=batch_norm)
        
    def forward(self, x):
        embedding = self.encoder(x)
        output = self.decoder(embedding)
        
        result = dict()
        result['bottleneck'] = embedding
        result['autoencoded'] = output
        return result