import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import label

# Encoding functions for found masks
def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list, verbose = 0):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    if verbose:
        print("Unique: {}".format(np.unique(all_masks)))
    all_masks[all_masks>0] = 1
    return np.expand_dims(all_masks, -1)

def show(x, y):
    f, axarr = plt.subplots(1,2, figsize=(15, 15))
    print (x.shape, y.shape)
    axarr[0].imshow(x.transpose(-1, 1, 0))
    axarr[1].imshow(y.transpose(-1, 1, 0)[:, :, 0])

def save_im(X, y, x_title, y_title, figname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
    ax1.imshow(X)
    ax1.set_title(x_title)
    ax2.imshow(y, cmap=plt.cm.gray)
    ax2.set_title(y_title)
    fig.savefig(fname=figname, bbox_inches = 'tight', pad_inches = 0)
    plt.close()


def image_pad(ip, pad_x_l=1, pad_x_r=1, pad_y_l=0, pad_y_r=0, constant_values=0):
        if (ip.shape[0] == 3):
            r = ip[0,:,:]
            g = ip[1,:,:]
            b = ip[2,:,:]
            r = np.pad(r, pad_width=((pad_x_l, pad_x_r), (pad_y_l, pad_y_r)), mode='constant', constant_values=constant_values)
            g = np.pad(g, pad_width=((pad_x_l, pad_x_r), (pad_y_l, pad_y_r)), mode='constant', constant_values=constant_values)
            b = np.pad(b, pad_width=((pad_x_l, pad_x_r), (pad_y_l, pad_y_r)), mode='constant', constant_values=constant_values)

            op = np.zeros((3, r.shape[0], r.shape[1]))
            op[0,:,:] = r
            op[1,:,:] = g
            op[2,:,:] = b
            return op

        elif (ip.shape[0] == 1):
            mask = np.pad(ip[0,:,:], pad_width=((pad_x_l, pad_x_r), (pad_y_l, pad_y_r)), mode='constant', constant_values=constant_values)
            op = np.zeros((1, mask.shape[0], mask.shape[1]))
            op[0, :, :] = mask
            return op

        else:
            print (np.shape)
            print ('Not an image')
            return ip