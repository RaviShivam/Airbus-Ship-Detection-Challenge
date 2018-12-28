import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import sigmoid
from torch.utils.data import Dataset, DataLoader


from tqdm import tqdm_notebook
import gc; gc.enable() # memory is tight

import traceback
import torchvision
import os


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
print ('asdfasdfadsf')

from skimage.io import imread
print ('asdfasdfadsf')

import matplotlib.pyplot as plt

print ('asdfasdfadsf')


from skimage.transform import rescale, resize, downscale_local_mean
from skimage import img_as_bool

from os.path import join
sys.path.append("../")
from src import utils

gc.enable()
torch.backends.cudnn.benchmark=True

print ('asdfasdfasdf')

DATA = "../data/"
MODELS = "../models"
MODEL_CHECKPOINT = join(MODELS, "UNETv1_checkpoint/")
TRAINING_STATS = join(MODEL_CHECKPOINT, "progress.csv")
TRAINED_UNET_MODEL = join(MODEL_CHECKPOINT, "model.pt")
TRAIN_PROGRESS_IMAGES = join(MODEL_CHECKPOINT, "progress/")

SHIP_DIR = "/media/shivam/DATA/airbus-tracking/"
TRAIN_IMAGE_DIR = os.path.join(SHIP_DIR, "train_v2")
TEST_IMAGE_DIR = os.path.join(SHIP_DIR, "test_v2")

CSV_TRAIN = join(DATA, 'balanced_train_df_shipgt_4.csv')
CSV_VALID = join(DATA, 'balanced_valid_df_shipgt_4.csv')


TRAIN_BATCH = 10
VALID_BATCH = 2


# # Dataset and preprocessing

# In[2]:


class KaggleDataset(Dataset):
    def __init__(self, datapath, phase):
        self.data = pd.read_csv(datapath)
        self.phase = phase
        self.image_ids = np.unique(self.data['ImageId'])
        print("Unique images: {}".format(len(self.image_ids)))
            
    def __len__(self):
        return len(self.image_ids)
    
    
    def __getitem__(self, idx):
        rgb_path = os.path.join(TRAIN_IMAGE_DIR, self.image_ids[idx])
        c_img = imread(rgb_path)
        c_mask = utils.masks_as_image(self.data[self.data['ImageId'] == self.image_ids[idx]]['EncodedPixels'].tolist())
        
        c_img = np.stack(c_img, 0)/255.0
        c_mask = np.stack(c_mask, 0)
        if self.phase == 'train':
            crop_delta = 192
            factor = 5
            h, w, _ = c_mask.shape
            # Random crop selection trick
            c = 0
            while (c < 5): 
                x1 = np.random.randint(0, h-crop_delta)
                x2 = np.random.randint(0, w-crop_delta)
                c_mask_s = c_mask[x1:x1+crop_delta, x2:x2+crop_delta, :]
                c += 1;
                if (np.sum(c_mask_s) > 200):
                    break
            c_img_s = c_img[x1:x1+crop_delta, x2:x2+crop_delta, :]
            c_img = c_img_s
            c_mask = c_mask_s

        c_img = c_img.transpose(-1, 0, 1)
        c_mask = c_mask.transpose(-1, 0, 1)
        return c_img.astype('f'), c_mask.astype('f')    

class Kaggle_Dataset_Test(Dataset):
    def __init__(self, testdir):
        self.testdir = testdir
        self.test_images = os.listdir(testdir)
        
    def __len__(self):
        return len(self.test_images)
    
    def __getitem__(self, idx):
        rgb_path = os.path.join(self.testdir, self.test_images[idx])
        c_img = imread(rgb_path)
        c_img = np.stack(c_img, 0)/255.0
        c_img = c_img_s.transpose(-1, 0, 1)
        return c_img


# # Unet
# In[3]:

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
    
def dice_loss(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return 1 - ((2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth))

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

def train(net, criterion, optimizer, epochs, trainLoader, valLoader):
    print ('Training has begun ...')
    training_stats = []
    for epoch in range(1, epochs+1):
        if (epoch > 1):
            plt.plot(range(epoch-1), np.array(training_stats)[:, 0], color='blue', marker='o', label='Train Loss')
            plt.plot(range(epoch-1), np.array(training_stats)[:, 1], color='orange', marker='o', label='Val Loss')
            plt.title('Loss Values (epoch : {0})'.format(epoch))
            plt.legend()
            plt.show()
        
        running_loss = 0
        val_loss = 0;
        # Train with all available data.
        print("Training in epoch: {}".format(epoch))
        tcount = 0
        for i, data in tqdm_notebook(enumerate(trainLoader), total = len(trainLoader)):
            if i > 15:
                break
            tcount += 1
            X,Y = data
            optimizer.zero_grad()

            # Y_   = net(X.cuda(non_blocking=True))
            Y_   = net(X.cuda(non_blocking=False))

            # loss = criterion(Y_, Y.cuda(non_blocking=True))
            loss = criterion(Y_, Y.cuda(non_blocking=False))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            del loss, X, Y, Y_  ## to be evaluated!!
        
        # Validate after each epoch.
        print("Starting validation...")
        val_loss = test(net, valLoader, criterion, save_im = True, group=epoch)
            
        # Normalize and save
        running_loss /= len(trainLoader)
        val_loss = sum(val_loss)/len(valLoader)
        training_stats.append([running_loss, val_loss])
        pd.DataFrame(training_stats).to_csv(TRAINING_STATS, header = ['running_loss', 'val_loss'], index = False)
        
        clear_output()
        
        # Save model
        if (epoch%5==0):
            print("Saving the model at {} epochs".format(epoch))
            torch.save(net.state_dict(), TRAINED_UNET_MODEL)
        
        # Empty gpu cache
        torch.cuda.empty_cache()
        print("Epoch: {}, running loss: {:.4f}, validation loss: {:.4f}".format(epoch, running_loss, val_loss))
        
        gc.collect()
        
def test(model, validLoader, criterion,  save_im = False, progress_path = TRAIN_PROGRESS_IMAGES, group=None):
    im_c = 0
    loss_scores = []
    with torch.no_grad():
        for i, data in tqdm_notebook(enumerate(validLoader), total=len(validLoader)):
            if i > 15:
                break
            X, y = data
            y_pred = model(X.cuda())
            loss_scores.append(criterion(y_pred, y.cuda()).item())

            # Save image if specified
            if save_im:
                for b in range(y_pred.shape[0]):

                    y_pred_tmp = y_pred[b, :, :, :].cpu().numpy().transpose(-1, 1, 0)[:, :, 0]
                    y_pred_tmp[y_pred_tmp >= 0.5] = 1
                    y_pred_tmp[y_pred_tmp < 0.5] = 0
                    
                    
                    X_tmp = X[b, :, :, :].numpy().transpose(-1, 1, 0)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
                    ax1.imshow(X_tmp)
                    ax1.set_title("Original image")
                    ax2.imshow(y_pred_tmp, cmap=plt.cm.gray)
                    ax2.set_title("Predicted segmentation")
                    figname = join(progress_path, "img{}_{}".format(im_c, group if group else "test"))
                    fig.savefig(fname=figname, bbox_inches = 'tight', pad_inches = 0)
                    plt.close()
                    im_c += 1
            
            del X,y, y_pred


    return loss_scores


# # Initializing for training
# In[4]:

# Load in Dataset
trainDataset = KaggleDataset(CSV_TRAIN, 'train')
validDataset = KaggleDataset(CSV_VALID, 'valid')

# Construct UNet
gc.collect()
reuse = False

net = UNet(3, 1).cuda()
if reuse:
    print("Reusing model from: {}".format(TRAINED_UNET_MODEL))
    net.load_state_dict(torch.load(TRAINED_UNET_MODEL))
    net.eval()
    
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
criterion = nn.BCELoss()

# Training the model
trainDataLoader   = torch.utils.data.DataLoader(
        trainDataset
        , batch_size=TRAIN_BATCH,shuffle=True
        , num_workers=0, pin_memory=True)

validDataLoader   = torch.utils.data.DataLoader(
        validDataset
        , batch_size=VALID_BATCH,shuffle=False
        , num_workers=0, pin_memory=True)


# # Training

# In[ ]:


train(net, criterion, optimizer, 20, trainDataLoader, validDataLoader)
