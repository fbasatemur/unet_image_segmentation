# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 11:24:53 2021

@author: fbasatemur
"""
import torch
from torch import nn

class Double_Conv(nn.Module):
    """double convulation => conv->relu->conv->relu"""

    def __init__(self, in_channels, out_channels, kernel_size, stride = (1,1), padding = 0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Up_Conv(nn.Module):
    """up convulation => upsampling->conv->relu"""

    def __init__(self, in_channels, out_channels, kernel_size = (2,2), stride = (2,2), padding = 0):
        super().__init__()
        self.up_conv = nn.Sequential(
            # nn.Upsample(scale_factor=kernel_size, mode='bilinear'),
            # nn.Conv2d(in_channels, out_channels, (1,1), (1,1), padding)
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        )

    def forward(self, x):
        return self.up_conv(x)
    
def Center_Crop(big_tensor, little_tensor_shape):
    cut_y, cut_x = (int((big_tensor.shape[2] - little_tensor_shape[2])/2), int((big_tensor.shape[3] - little_tensor_shape[3])/2))
    return big_tensor[:,:,cut_y: cut_y + little_tensor_shape[2], cut_x: cut_x + little_tensor_shape[3]]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.maxpool = nn.MaxPool2d((2,2))
        self.drop = nn.Dropout2d(0.5)     

        self.conv1 = Double_Conv(1, 64, 3)
        self.conv2 = Double_Conv(64, 128, 3)
        self.conv3 = Double_Conv(128, 256, 3)
        self.conv4 = Double_Conv(256, 512, 3)  

        self.conv5 = Double_Conv(512, 1024, 3)
        
        self.up_conv1 = Up_Conv(1024, 512)
        # concat conv4 + up_conv1 
        self.conv6 = Double_Conv(512 + 512, 512, 3) 

        self.up_conv2 = Up_Conv(512, 256)
        # concat conv3 + up_conv2
        self.conv7 = Double_Conv(256 + 256, 256, 3) 

        self.up_conv3 = Up_Conv(256, 128)
        # concat conv2 + up_conv3
        self.conv8 = Double_Conv(128 + 128, 128, 3) 

        self.up_conv4 = Up_Conv(128, 64)
        # concat conv1 + up_conv4
        self.conv9 = Double_Conv(64 + 64, 64, 3)
        
        self.conv10 = nn.Conv2d(64, 1, 1) 

    def forward(self, x):
        x_conv1 = self.conv1(x)
        #print(x_conv1.shape)
        x = self.maxpool(x_conv1)
        #print(x.shape)

        x_conv2 = self.conv2(x)
        #print(x_conv2.shape)
        x = self.maxpool(x_conv2)
        #print(x.shape)

        x_conv3 = self.conv3(x)
        #print(x_conv3.shape)
        x = self.maxpool(x_conv3)
        #print(x.shape)

        x = self.conv4(x)
        x_drop4 = self.drop(x)
        #print(x_drop4.shape)
        x = self.maxpool(x_drop4)
        #print(x.shape)
        
        x = self.conv5(x)
        x = self.drop(x)
        #print(x.shape)

        x = self.up_conv1(x)
        #print(x.shape)
        x = torch.cat((Center_Crop(x_drop4, x.shape), x), dim=1)
        #print(x.shape)
        x = self.conv6(x)
        #print(x.shape)

        x = self.up_conv2(x)
        #print(x.shape)
        x = torch.cat((Center_Crop(x_conv3, x.shape), x), dim=1)
        #print(x.shape)
        x = self.conv7(x)
        #print(x.shape)
        
        x = self.up_conv3(x)
        #print(x.shape)
        x = torch.cat((Center_Crop(x_conv2, x.shape), x), dim=1)
        #print(x.shape)
        x = self.conv8(x)
        #print(x.shape)

        x = self.up_conv4(x)
        #print(x.shape)
        x = torch.cat((Center_Crop(x_conv1, x.shape), x), dim=1)
        #print(x.shape)
        x = self.conv9(x)        
        #print(x.shape)
        
        output = torch.sigmoid(self.conv10(x))
        #print(output.shape)

        return output
    
    