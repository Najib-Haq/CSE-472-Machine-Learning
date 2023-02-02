import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import math

from dataset.augments import rotate, blur, get_number_bb

class Dataset:
    def __init__(self, directory, df, label_col, config, mode='train'):
        '''
        directory = parent directory of the dataset
        '''
        self.directory = directory
        self.df = df
        self.label_col = label_col
        self.mode = mode
        self.config = config

    def __len__(self):
        return len(self.df)

    def augment(self, image):
        if np.random.rand() < 0.5:
            image = rotate(image, -10, 10)
        if np.random.rand() < 0.5:
            image = blur(image)
        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(os.path.join(self.directory, row['database name'] + '/' + row['filename']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # use only bounding box
        if self.config['use_bbox']: image = get_number_bb(image)
        # reverse
        if self.config['reverse']: image = 255 - image
        # use probabilistic augmentation
        if self.config['aug'] and self.mode == 'train': image = self.augment(image)

        # resize and normalize
        image = cv2.resize(image, (self.config['img_shape'][0], self.config['img_shape'][1]), interpolation = cv2.INTER_AREA)
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        
        if self.mode in ["train", "valid"] : return image, row[self.label_col]
        else: return image


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.idx = 0
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            print("Shuffling Dataset. ")
            self.dataset.df = self.dataset.df.sample(frac=1).reset_index(drop=True)
        return self

    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)

    def __next__(self):
        if self.idx >= len(self.dataset):
            raise StopIteration


        batch = []       
        for i in range(self.batch_size):
            if self.idx >= len(self.dataset):
                break
            data = self.dataset[self.idx]
            if self.dataset.mode in ["train", "valid"]: 
                if len(batch) == 0: batch = [[], []]
                batch[0].append(data[0])
                batch[1].append(data[1])
            else: batch.append(data)
            self.idx += 1

        if self.dataset.mode in ["train", "valid"]: batch = [np.stack(batch[0]), np.array(batch[1])]
        else: batch = np.stack(batch)
        return batch


    
def check_dataset(train_dataset, valid_dataset, save_dir):
    train_idx = np.random.randint(0, len(train_dataset))
    valid_idx = np.random.randint(0, len(valid_dataset))

    train_image, train_label = train_dataset[train_idx]
    valid_image, valid_label = valid_dataset[valid_idx]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(train_image.transpose(1, 2, 0))
    ax[0].set_title(f"Train[{train_idx}]: {train_label}")
    ax[1].imshow(valid_image.transpose(1, 2, 0))
    ax[1].set_title(f"Valid[{valid_idx}]: {valid_label}")

    # save as image
    fig.savefig(f'{save_dir}/dataset.png', dpi=300, bbox_inches='tight')
    # plt.show()


