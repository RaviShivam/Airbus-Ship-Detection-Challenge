import gc; gc.enable() 
import torch
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython import get_ipython

from IPython.display import clear_output
from tqdm import tqdm_notebook
from os.path import join

progressbar = tqdm_notebook if get_ipython() else tqdm


class DeepLearning:
    def __init__(self, net, model_checkpoint, cuda=True):
        self.net = net
        self.model_checkpoint = model_checkpoint
        self.training_log = join(self.model_checkpoint, "progress.csv")
        self.training_plot = join(self.model_checkpoint, "progress.png")
        self.trained_model = join(self.model_checkpoint, "model.pt")
        self.training_progress_images = join(self.model_checkpoint, "progress/")
        self.cuda = cuda
        if not os.path.exists(self.model_checkpoint): 
            os.mkdir(model_checkpoint)
        if not os.path.exists(self.training_progress_images): 
            os.mkdir(self.training_progress_images)

    def __save_progress_images__(self, epoch):
        pass 

    def __show_progress__(self, epoch, training_stats):
        if (epoch > 1):
            plt.plot(range(epoch-1), np.array(training_stats)[:, 0], color='blue', marker='o', label='Train Loss')
            plt.plot(range(epoch-1), np.array(training_stats)[:, 1], color='orange', marker='o', label='Val Loss')
            plt.title('Loss Values (epoch : {0})'.format(epoch))
            plt.legend()
            plt.savefig(fname=self.training_plot, bbox_inches = 'tight', pad_inches = 0)
            if get_ipython():
                plt.show()

    def fit(self, trainLoader, epochs, optimizer, criterion, valLoader=None, save_learning_progress=True):
        print ('Training has begun ...')

        training_stats = []
        for epoch in range(1, epochs+1):
            self.__show_progress__(epoch, training_stats)
            
            # Reinitialize epoch scores.
            running_loss = 0
            val_loss = 0

            # Train with all available data.
            print("Training in epoch: {}".format(epoch))
            for i, data in progressbar(enumerate(trainLoader), total = len(trainLoader)):
                if i > 2:
                    break
                # Initialize
                X,Y = data
                optimizer.zero_grad()

                # Predict and determine loss
                Y_   = self.net(X.cuda(non_blocking=False)) if self.cuda else self.net(X)
                loss = criterion(Y_, Y.cuda(non_blocking=False)) if self.cuda else criterion(Y_, Y)
                running_loss += loss.item()

                # Backpropagation
                loss.backward()
                optimizer.step()
                
                # Clear memory
                del loss, X, Y, Y_  
            
            # Validate after each epoch.
            print("Starting validation...")
            val_loss = self.validate(valLoader, criterion, save_im = True, group=epoch) if valLoader else 0
                
            # Normalize and save learning progress
            running_loss /= len(trainLoader)
            val_loss = sum(val_loss)/len(valLoader)
            training_stats.append([running_loss, val_loss])
            pd.DataFrame(training_stats).to_csv(self.training_log, header = ['running_loss', 'val_loss'], index = False)
            
            # Clean system out.
            clear_output() if epoch < epochs else None
            
            # Save model
            if (epoch%5==0):
                print("Saving the model at {} epochs".format(epoch))
                torch.save(self.net.state_dict(), self.trained_model)
            
            # Empty gpu cache.
            torch.cuda.empty_cache() if self.cuda else None
            gc.collect()

            # Log progress.
            print("Epoch: {}, running loss: {:.4f}, validation loss: {:.4f}".format(epoch, running_loss, val_loss))

    def save_im(X, y):
        pass
        
    def validate(self, valLoader, criterion, save_im=True, group=None):
        im_c = 0
        loss_scores = []

        with torch.no_grad():
            for i, data in progressbar(enumerate(valLoader), total=len(valLoader)):
                if i > 2:
                    break
                # Predict input
                X, y = data
                y_pred = self.net(X.cuda()) if self.cuda else self.net(X)

                # Save progress
                loss_current = criterion(y_pred, y.cuda()).item() if self.cuda else criterion(y_pred, y).item()
                loss_scores.append(loss_current)

                # Save image if specified
                if save_im:
                    for b in range(y_pred.shape[0]):

                        y_pred_tmp = y_pred[b, :, :, :].cpu().numpy()
                        figname = join(self.training_progress_images, "img{}_{}".format(im_c, group if group else "test"))

                        Xnp = X[b, :, :, :].cpu().numpy()
                        valLoader.dataset.convertScoreMapToImg(Xnp, y_pred_tmp, figname)

                        im_c += 1
                
                del X,y, y_pred
        return loss_scores


    def predict(self, testDataLoader):
        map_location = torch.device('gpu') if self.cuda else torch.device('cpu')
        self.net.load_state_dict(torch.load(self.trained_model, map_location=map_location))
        self.net.eval()
        out_pred_rows = []
        with torch.no_grad():
            for i, data in progressbar(enumerate(testDataLoader), total=len(testDataLoader)):
                if i>5:
                    break
                c_img_name, X = data
                y_pred = self.net(X.cuda()) if self.cuda else self.net(X)
                
                for b in range(y_pred.shape[0]):
                    y_pred_tmp = y_pred[b, :, :, :].cpu().numpy()
                    out_pred_rows += testDataLoader.dataset.item_to_encoding(c_img_name[b], y_pred_tmp)
                    gc.collect()
        return out_pred_rows