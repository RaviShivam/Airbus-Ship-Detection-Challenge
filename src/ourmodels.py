import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import sigmoid
from torchvision import models

class double_conv(nn.Module):
    ''' conv -> BN -> relu -> conv -> BN -> relu'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
    
    def forward(self, x):
        return self.conv(x)

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )
    
    def forward(self, x):
        return self.mpconv(x)

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, out_ch//2, stride=2)
        
        self.conv = double_conv(in_ch, out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, (diffX//2, diffX - diffX//2,
                        diffY//2, diffY - diffY//2)
                  )
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64,  128) #x2
        self.down2 = down(128, 256) #x3
        self.down3 = down(256, 512) #x4
        self.down4 = down(512, 512) #x5
        self.up1   = up(1024,256)
        self.up2   = up(512,128)
        self.up3   = up(256,64)
        self.up4   = up(128,64)
        self.outc  = outconv(64, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5,x4) # (x5-512d + x4-512d  = 1024d--> 256d)
        x = self.up2(x,x3)  # (x-256d + x3 - 256d = 512d --> 128d)
        x = self.up3(x, x2) # (x-128d + x2 - 128d = 256d --> 64d)
        x = self.up4(x,x1)  # (x-64d  + x1 - 64d  = 128d --> 64d)
        x = self.outc(x)    # 64d --> n_classes_D
        
        return sigmoid(x)


class ResNetUnet(nn.Module):
    
    def __init__(self, n_classes, verbose=0):
        super().__init__()
        def convrelu(in_channels, out_channels, kernel, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
                nn.ReLU(inplace=True)
            )
        
        self.verbose     = verbose 
        
        self.base_model  = models.resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())
        
        self.layer0      = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1  = convrelu(64,64,1,0)
        
        self.layer1      = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1  = convrelu(64,64,1,0)
        
        self.layer2      = self.base_layers[5]
        # self.layer2_1x1  = convrelu(512, 512, 1,0)
        self.layer2_1x1  = convrelu(128, 64, 1,0)
        
        self.layer3      = self.base_layers[6]
        # self.layer3_1x1  = convrelu(1024,512,1,0)
        self.layer3_1x1  = convrelu(256,128,1,0)
        
        self.layer4      = self.base_layers[7]
        # self.layer4_1x1  = convrelu(2048, 1024,1,0)
        self.layer4_1x1  = convrelu(512, 256,1,0)
        
        self.upsample    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # self.conv_up3    = convrelu(512+1024 , 512, 3, 1)
        self.conv_up3    = convrelu(128+256 , 128, 3, 1)
        #self.conv_up2    = convrelu(512 + 512, 512, 3, 1) 128+64
        self.conv_up2    = convrelu(128+64, 64, 3, 1)
        #self.conv_up1    = convrelu(256 + 512, 256, 3, 1)
        self.conv_up1    = convrelu(64 + 64, 64, 3, 1)
        # self.conv_up0    = convrelu(64 + 256 , 128, 3, 1)
        self.conv_up0    = convrelu(64 + 64 , 64, 3, 1)
        
        self.conv_original_size0 = convrelu(3, 32, 3, 1)
        self.conv_original_size1 = convrelu(32, 64, 3, 1)
        self.conv_original_size2 = convrelu(128, 64, 3, 1)
        
        self.conv_last   = nn.Conv2d(64, n_classes, 1)
    
    def forward(self, x_input):
        if self.verbose: print ('Preprocess : x_input    : ', x_input.shape)
        x_original = self.conv_original_size0(x_input)
        if self.verbose: print ('Preprocess : x_original : ', x_original.shape)
        x_original = self.conv_original_size1(x_original)
        if self.verbose: print ('Preprocess : x_original : ', x_original.shape)
        
        
        
        layer0 = self.layer0(x_input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        
        if self.verbose:
            print ('\nEncoder : Layer0 : ', layer0.shape)
            print ('Encoder : Layer1 : ', layer1.shape)
            print ('Encoder : Layer2 : ', layer2.shape)
            print ('Encoder : Layer3 : ', layer3.shape)
            print ('Encoder : Layer4 : ', layer4.shape)
        
        layer4 = self.layer4_1x1(layer4)
        if self.verbose: print ('\nDecoder : Layer4 : ', layer4.shape)
        
        x      = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x      = torch.cat([x, layer3], dim=1)
        x      = self.conv_up3(x)
        if self.verbose: print ('Decoder : Layer3 : ', x.shape)
                
        x      = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x      = torch.cat([x, layer2], dim=1)
        x      = self.conv_up2(x)
        if self.verbose: print ('Decoder : Layer2 : ', x.shape)
        
        x      = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x      = torch.cat([x, layer1], dim=1)
        x      = self.conv_up1(x)
        if self.verbose: print ('Decoder : Layer1 : ', x.shape) # [1, 64, h/4, w/4]
        
        x      = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x      = torch.cat([x, layer0], dim=1)
        x      = self.conv_up0(x)
        if self.verbose:print ('Decoder : Layer0 : ', x.shape) # [1, 64, h/2, w/2]
        
        x      = self.upsample(x)
        x      = torch.cat([x, x_original], dim=1)
        x      = self.conv_original_size2(x)
        if self.verbose : print ('Decoder : x    : ', x.shape) 
        
        out    = self.conv_last(x)
        if self.verbose : print ('Decoder : out  : ', out.shape)
        
        return F.sigmoid(out)
        