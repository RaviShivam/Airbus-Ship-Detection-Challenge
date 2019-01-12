import pandas as pd 
import numpy as np
import os
from os.path import join 
from sklearn.model_selection import train_test_split

DATA = "./"
SHIP_DIR = "/media/shivam/DATA/airbus-tracking/"
TRAIN_IMAGE_DIR = os.path.join(SHIP_DIR, "train_v2")

def preprocess(use_csv, min_ship_count):
    
    def sample_ships(in_df, base_rep_val=1500):
        if in_df['ships'].values[0]==0:
            return in_df.sample(base_rep_val//3) # even more strongly undersample no ships
        else:
            return in_df.sample(base_rep_val, replace=(in_df.shape[0]<base_rep_val))
    
    if use_csv: 
        train_df = pd.read_csv(join(DATA, 'balanced_train_df.csv'))
        valid_df = pd.read_csv(join(DATA, 'balanced_valid_df.csv'))
            
    else:
        masks = pd.read_csv(os.path.join(SHIP_DIR, 'train_ship_segmentations_v2.csv'))

#         masks = masks.sample(len(masks)/2)
        masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
        unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
        unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
        unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
        
        # some files are too small/corrupt
        print("Removing small files")
        unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id: 
                                                                        os.stat(os.path.join(TRAIN_IMAGE_DIR, 
                                                                                            c_img_id)).st_size/1024)
        unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb']>50] # keep only 50kb files
        masks.drop(['ships'], axis=1, inplace=True)
        train_ids, valid_ids = train_test_split(unique_img_ids, 
                            test_size = 0.01, 
                            stratify = unique_img_ids['ships'])


        # train_df = pd.merge(masks, train_ids) ## on='ImageId'
        # valid_df = pd.merge(masks, valid_ids) ## on='ImageId'

        train_df = pd.merge(masks, train_ids, on='ImageId')
        valid_df = pd.merge(masks, valid_ids, on='ImageId')

        # print("Grouping ships")
        # train_df['grouped_ship_count'] = train_df['ships'].map(lambda x: (x+1)//2)
        # valid_df['grouped_ship_count'] = valid_df['ships'].map(lambda x: (x+1)//2)
        
        # balanced_train_df = train_df.groupby('grouped_ship_count').apply(sample_ships)
        # balanced_valid_df = valid_df.groupby('grouped_ship_count').apply(sample_ships)

        # balanced_train_df = balanced_train_df.groupby('ImageId')
        # balanced_valid_df = balanced_valid_df.groupby('ImageId')


        train_df.to_csv(join(DATA, "balanced_train_df.csv"), index=False)
        valid_df.to_csv(join(DATA, "balanced_valid_df.csv"), index=False)

        
    # TRAINING 
    filename_train = join(DATA, 'balanced_train_df_shipgt_{0}.csv'.format(min_ship_count))
    if not os.path.exists(filename_train):
        print("Creating new csv files: {}".format(filename_train))
        ourBalanced_train_df = train_df[train_df['ships'] >= min_ship_count]
        print("Original lenght: {}, result length : {}".format(len(train_df), len(ourBalanced_train_df)))
        ourBalanced_train_df.to_csv(filename_train, index=False)
    
    # VALIDATION
    filename_validation = join(DATA, 'balanced_valid_df_shipgt_{0}.csv'.format(min_ship_count))  
    if not os.path.exists(filename_validation):
        print("Creating new csv files: {}".format(filename_validation))
        ourBalanced_valid_df = valid_df[valid_df['ships'] >= min_ship_count]
        print("Original lenght: {}, result length : {}".format(len(valid_df), len(ourBalanced_valid_df)))
        ourBalanced_valid_df.to_csv(filename_validation, index=False)


preprocess(False, 4)
        
# def validationset(size=10, batch_size=2):
#     random_batches = [np.random.randint(0, len(all_batches_balancedValid)-batch_size) for _ in range(10)]
#     for i in random_batches:
#         X = []
#         Y = []
#         for j in range(batch_size):
#             X_temp, y_temp = extract_image(i+j, train_image_dir, all_batches_balancedValid)
#             X.append(X_temp)
#             Y.append(y_temp)
#         X = np.array(X)
#         Y = np.array(Y)
#         yield X, Y

# def testset(path= TEST_IMAGE_DIR, batch = 2):
#     test_images = os.listdir(path)
#     if testdata:
#         c_path = join(c_img_name)
#         c_img = imread(c_path)
#         c_img = c_img.transpose(-1, 0, 1).astype('f')
#         c_img = np.expand_dims(c_img, 0)/255.0
#     return testdata

# def show(x, y):
#     f, axarr = plt.subplots(1,2, figsize=(15, 15))

#     axarr[0].imshow(x.transpose(-1, 1, 0))
#     axarr[1].imshow(y.transpose(-1, 1, 0)[:, :, 0])

# def extract_image(idx, datapath, data):
#     rgb_path = os.path.join(datapath, data[idx][0])
#     c_img = imread(rgb_path)
#     c_mask = masks_as_image(data[idx][1]['EncodedPixels'].values)
    
#     c_img = c_img.transpose(-1, 0, 1)
#     c_mask = c_mask.transpose(-1, 0, 1)
#     return c_img.astype('f'), c_mask.astype('f')