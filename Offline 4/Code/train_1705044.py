import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import math
import pickle
import yaml
from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from scipy import ndimage


# wandb stuff
try:
    import wandb
except:
    print("please install wandb if you want to use wandb loggings")

###################################################################################################################
###################################################################################################################
###################################################################################################################
# dataset/augments.py
def rotate(image, angleFrom=-10, angleTo=10):
    '''
    angles in degrees
    '''
    angle = np.random.randint(angleFrom, angleTo)
    return ndimage.rotate(image, angle)


def blur(image):
    sigma = np.random.randint(3, 6)
    return ndimage.gaussian_filter(image, sigma=sigma)


def get_contour_cutout(image, contours, contour_cutout_number):
    image = image.copy()
    no_contours =contours.shape[0]
    # choose random number from 0 to contour_cutout_number
    contour_cutout_number_sample = np.random.randint(2, contour_cutout_number)
    # choose random contours 
    random_contours = np.random.randint(0, no_contours, contour_cutout_number_sample)

    height, width, _ = image.shape
    cutout_height, cutout_width = height*0.02, width*0.02
    for point in contours[random_contours]:
        x, y = point[0][0], point[0][1]
        # cutout the image
        # also check whether the cutout is within the image
        if (x-cutout_width) > 0 and (x+cutout_width) < width and (y-cutout_height) > 0 and (y+cutout_height) < height:
            image[int(y-cutout_height):int(y+cutout_height), int(x-cutout_width):int(x+cutout_width), :] = 0

    return image


def get_number_bb(image, use_bbox,):
    image = image.copy()
    # convert to grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (_, binary) = cv2.threshold(gray, 255//2, 255, cv2.THRESH_OTSU)
    binary = 255-binary

    # apply erosion + dilation to remove noise
    kernel = np.ones((5,5),np.uint8)
    img_opening = cv2.erode(cv2.dilate(binary,kernel,iterations = 1), kernel,iterations = 1)
    # plt.figure()
    # plt.imshow(img_opening, cmap='gray')

    # get bounding box
    contours, _ = cv2.findContours(img_opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))[-1]

    
    if use_bbox:
        bounding_box = cv2.boundingRect(contours)
        return bounding_box, contours
    return _, contours


###################################################################################################################
###################################################################################################################
###################################################################################################################
# dataset/dataset.py

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

        self.cache = self.config['cache']
        self.cache_data = {}
        self.cache_contour = {}
        self.cache_bbox = {}
        if self.cache: self.cache_image()

    def __len__(self):
        return len(self.df)
    
    def change_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # opening
        if self.config['opening']: image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        
        # use only bounding box
        if self.config['use_bbox'] or (self.config['contour_cutout_prob'] > 0.0): 
            bounding_box, contours = get_number_bb(image, self.config['use_bbox'])
            self.cache_contour[path] = contours
            self.cache_bbox[path] = bounding_box
        # reverse
        if self.config['reverse']: image = 255 - image
        # dilation
        if self.config['dilation']: image = cv2.dilate(image, np.ones((5, 5), np.uint8), iterations = 1)
        
        # resize
        # if not train need to apply these now as no contour cutout in valid/test
        if self.mode != 'train' or (self.config['contour_cutout_prob'] == 0.0):
            if self.config['use_bbox']:
                bounding_box = self.cache_bbox[path]
                image = image[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2], :]

            # need to apply contour before resize so this condition is applied
            image = cv2.resize(image, (self.config['img_shape'][0], self.config['img_shape'][1]), interpolation = cv2.INTER_AREA)

        return image
    
    def cache_image(self):
        print("Cache Dataset...")
        for i in tqdm(range(len(self.df))):
            row = self.df.iloc[i]
            path = os.path.join(self.directory, row['img_path'])
            self.cache_data[path] = self.change_image(path)

    def augment(self, image):
        if np.random.rand() < 0.5:
            image = rotate(image, -10, 10)
        if np.random.rand() < 0.5:
            image = blur(image)
        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.directory, row['img_path'])

        if self.cache:
            image = self.cache_data[path]
        else:
            image = self.change_image(path)

        # use probabilistic augmentation
        if self.config['aug'] and self.mode == 'train': image = self.augment(image)

        # use contour cutout
        if self.mode == 'train':
            # need to apply contour before resize and/or bbox so this condition is applied
            if (self.config['contour_cutout_prob'] > np.random.rand()):
                image = get_contour_cutout(image, self.cache_contour[path], self.config['contour_cutout_number'])
            if self.config['use_bbox']:
                bounding_box = self.cache_bbox[path]
                image = image[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2], :]
            image = cv2.resize(image, (self.config['img_shape'][0], self.config['img_shape'][1]), interpolation = cv2.INTER_AREA)
            
        # resize and normalize
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        
        if self.mode in ["train", "valid"] : return image, row[self.label_col]
        else: return image, row['img_path']


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
            else: 
                if len(batch) == 0: batch = [[], []]
                batch[0].append(data[0])
                batch[1].append(data[1])
            self.idx += 1

        if self.dataset.mode in ["train", "valid"]: batch = [np.stack(batch[0]), np.array(batch[1])]
        else: batch = [np.stack(batch[0]), batch[1]]
        return batch


    
def check_dataset(train_dataset, valid_dataset, save_dir, from_mixup=False):
    train_idx = np.random.randint(0, len(train_dataset))
    valid_idx = np.random.randint(0, len(valid_dataset))

    if from_mixup:
        train_image = train_dataset[0][train_idx]
        train_label = train_dataset[1][train_idx]
    else:
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

def check_test_dataset(test_dataset, save_dir):
    test_idx = np.random.randint(0, len(test_dataset))
    test_image, path = test_dataset[test_idx]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(test_image.transpose(1, 2, 0))
    ax.set_title(f"Test[{test_idx}]")

    # save as image
    fig.savefig(f'{save_dir}/dataset_test.png', dpi=300, bbox_inches='tight')
    # plt.show()


###################################################################################################################
###################################################################################################################
###################################################################################################################
# model/nn/Base.py
class Base:
    def __init__(self):
        self.params = {}
        self.state_dict = {}
        self.grads = {}
        self.cache = {} # keep information from forward pass needed for backward pass
        self.name = "Base"
        self.trainable = True

    def __str__(self):
        return self.name + "(" + str(self.params) + ")"

    def forward(self, X):
        pass

    def __call__(self, X):
        return self.forward(X)

    def update_weights(self, lr):
        if self.trainable:
            for key in self.state_dict:
                self.state_dict[key] -= lr * self.grads[key]
        # print(self.name + " : " , self.state_dict)

###################################################################################################################
###################################################################################################################
###################################################################################################################
# model/nn/Conv2d.py

class Conv2D(Base):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.name = "Conv2D"
        self.params = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "bias": bias,
        }
        self.state_dict = self.initialize_parameters()

    def initialize_parameters(self):
        # He initialization
        # https://stackoverflow.com/questions/48641192/xavier-and-he-normal-initialization-difference#:~:text=In%20summary%2C%20the%20main%20difference,for%20layers%20with%20sigmoid%20activation.
        std = np.sqrt(2 / (self.params["in_channels"] * self.params["kernel_size"] * self.params["kernel_size"]))
        # channel first shape
        kernels = np.random.randn(self.params["out_channels"], self.params["in_channels"], self.params["kernel_size"], self.params["kernel_size"]) * std
        if self.params["bias"]:
            bias = np.zeros(self.params["out_channels"])
            return {"kernels": kernels, "bias": bias}
        return {"kernels": kernels}

    def generate_strided_tensor(self, X, kernel_size, stride, padding, out_shape, strides=None):
        '''
        here kernel_size, stride, padding are tuples of (H, W)
        ''' 
        C_out = X.shape[1]
        N, _, H_out, W_out = out_shape

        # pad the input tensor if necessary
        if padding != (0, 0):
            X = np.pad(X, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode="constant", constant_values=0)
        
            
        # get strides of X
        N_strides, C_out_strides, H_strides, W_strides = X.strides if strides is None else strides
        # create a strided tensor
        # use this link to understand: https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20
        strided_tensor = np.lib.stride_tricks.as_strided(
            X, 
            shape=(N, C_out, H_out, W_out, kernel_size[0], kernel_size[1]), 
            strides=(N_strides, C_out_strides, stride[0] * H_strides, stride[1] * W_strides, H_strides, W_strides)
        )
        return strided_tensor


    def forward(self, X):
        '''
        X shape should be (N, C, H, W)
        '''

        kernel_size, stride, padding, bias = self.params["kernel_size"], self.params["stride"], self.params["padding"], self.params["bias"]
        # get output shape
        B, C_in, H_in, W_in = X.shape
        H_out = (H_in + 2 * padding - kernel_size) // stride + 1
        W_out = (W_in + 2 * padding - kernel_size) // stride + 1
        output_shape = (B, C_in, H_out, W_out)

        # get strided X windows
        strided_X = self.generate_strided_tensor(X, (kernel_size, kernel_size), (stride, stride), (padding, padding), output_shape)

        # convolution with kernels
        # use this to understand: https://ajcr.net/Basic-guide-to-einsum/
        output = np.einsum("nchwkl,ockl->nohw", strided_X, self.state_dict["kernels"])

        # add bias if necessary
        if bias:
            output += self.state_dict["bias"][np.newaxis, :, np.newaxis, np.newaxis]  
        
        if self.trainable:
            self.cache = {
                "strided_X": strided_X,
                "X_shape": (B, C_in, H_in, W_in)
            }

        return output

    def backward(self, dL_dy):
        '''
        dL_dy = gradient of the cost with respect to the output of the conv layer -> (bs, C_out, H, W)

        compute :
        dL_dK = gradient of the cost with respect to the kernels -> (C_out, C_in, kernel_size, kernel_size)
        dL_db = gradient of the cost with respect to the bias -> (C_out)
        dL_dX = gradient of the cost with respect to the input -> (bs, C_in, H_in, W_in)

        '''
        # get parameters
        kernel_size, stride, padding, bias = self.params["kernel_size"], self.params["stride"], self.params["padding"], self.params["bias"]

        # compute dL_dK and dL_db
        dL_dK = np.einsum("nchwkl,nohw->ockl", self.cache['strided_X'], dL_dy) # Convolution(X, dL_dy)
        if bias:
            dL_db = np.einsum('nohw->o', dL_dy) # sum over N, H, W

        # compute dL_dX
        # rotate kernels 180
        kernels_rotated = np.rot90(self.state_dict["kernels"], k=2, axes=(2, 3)) # (number of times rotated by 90, k=2)
        # get strided dL_dy windows
        dout_padding = kernel_size - 1 if padding == 0 else kernel_size - 1 - padding
        # dout_padding = 1
        dout_dilate = stride - 1
        # dilate dL_dy based on stride
        if dout_dilate != 0:
            insertion_indices = list(np.arange(1, dL_dy.shape[2]))*dout_dilate
            # print(insertion_indices)
            dL_dy_dilated = np.insert(dL_dy, insertion_indices, values=0, axis=2) # args - input, index, value, axis
            dL_dy_dilated = np.insert(dL_dy_dilated, insertion_indices, values=0, axis=3)

            # in cases where right and bottom gets ignored, these can be added back by extra padding
            new_shape_h = (dL_dy_dilated.shape[2] + 2 * dout_padding - kernel_size) // 1 + 1
            new_shape_w = (dL_dy_dilated.shape[3] + 2 * dout_padding - kernel_size) // 1 + 1
            if (new_shape_h != self.cache['X_shape'][2]) or (new_shape_w != self.cache['X_shape'][3]):
                 # pad incase of odd shape
                pad_h = self.cache['X_shape'][2] - new_shape_h 
                pad_w = self.cache['X_shape'][3] - new_shape_w 
                dL_dy_dilated = np.pad(dL_dy_dilated, ((0,0), (0,0), (0,pad_h), (0,pad_w)))

        else:
            dL_dy_dilated = dL_dy.copy()
        strided_dL_dy = self.generate_strided_tensor(dL_dy_dilated, (kernel_size, kernel_size), (1, 1), (dout_padding, dout_padding), self.cache['X_shape'], strides=None)
        # compute dL_dX
        dL_dX = np.einsum("nohwkl,ockl->nchw", strided_dL_dy, kernels_rotated) # Convolution(padded dL_dy, kernels_rotated)

        # update parameters
        self.grads['kernels'] = dL_dK
        if bias: self.grads['bias'] = dL_db

        return dL_dX
 

###################################################################################################################
###################################################################################################################
###################################################################################################################

# model/nn/Maxpool2d

class MaxPool2D(Base):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.name = "MaxPool2D"
        self.params = {
            "kernel_size": kernel_size,
            "stride": stride,
        }
        self.cache = {}
        self.same_kernel_stride = kernel_size == stride  # fully vectorize when kernel_size == stride

    def forward(self, X):
        N, C, H, W = X.shape
        if self.trainable:
            self.cache['X_shape'] = X.shape
            self.cache['X_strides'] = X.strides
        kernel_size, stride = self.params["kernel_size"], self.params["stride"]

        # output shape
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1

        # get kernel strided X
        N_strides, C_out_strides, H_strides, W_strides = X.strides
        strided_X = np.lib.stride_tricks.as_strided(
            X,
            shape=(N, C, H_out, W_out, kernel_size, kernel_size),
            strides=(N_strides, C_out_strides, stride * H_strides, stride * W_strides, H_strides, W_strides)
        )

        # max pooling
        output = np.max(strided_X, axis=(4, 5))
        if self.trainable: 
            if self.same_kernel_stride: 
                maxes_reshaped_to_original_window = output.repeat(stride, axis=-2).repeat(stride, axis=-1)
                # pad incase of odd shape
                pad_h = H - maxes_reshaped_to_original_window.shape[-2]
                pad_w = W - maxes_reshaped_to_original_window.shape[-1]
                maxes_reshaped_to_original_window = np.pad(maxes_reshaped_to_original_window, ((0,0), (0,0), (0,pad_h), (0,pad_w)))
                self.cache['mask'] = np.equal(X, maxes_reshaped_to_original_window)
            else: self.cache['strided_X'] = strided_X
        return output

    def fully_vectorized_backward(self, dL_dy):
        # not that much increase :/
        # https://stackoverflow.com/questions/61954727/max-pooling-backpropagation-using-numpy
        stride = self.params["stride"]
        N, C, H, W = self.cache['X_shape']
        dL_dy_reshaped_to_original_window = dL_dy.repeat(stride, axis=-2).repeat(stride, axis=-1)
        
        # pad incase of odd shape
        pad_h = H - dL_dy_reshaped_to_original_window.shape[-2]
        pad_w = W - dL_dy_reshaped_to_original_window.shape[-1]
        dL_dy_reshaped_to_original_window = np.pad(dL_dy_reshaped_to_original_window, ((0,0), (0,0), (0,pad_h), (0,pad_w)))
        
        dL_dy_reshaped_to_original_window = np.multiply(dL_dy_reshaped_to_original_window, self.cache['mask'])
        return dL_dy_reshaped_to_original_window

    def partially_vectorized_backward(self, dL_dy):
        N, C, H_out, W_out = dL_dy.shape
        kernel_size, stride = self.params["kernel_size"], self.params["stride"]

        # get cached strided_X
        strided_X = self.cache['strided_X']

        reshaped_strided_X = strided_X.reshape((N, C, H_out, W_out, -1)) # need to do this as cannot get max from multiple axis
        argmaxes = reshaped_strided_X.argmax(axis=-1)
        a1, a2, a3, a4 = np.indices((N, C, H_out, W_out)) # indices of axies
        argmaxes_indices = (a1, a2, a3, a4, argmaxes)
        
        # set to 1 and then multiply with gradient
        strided_X_maxes = np.zeros_like(reshaped_strided_X)
        strided_X_maxes[argmaxes_indices] = 1
        strided_X_maxes *= dL_dy[..., None]

        # reshape to original shape
        strided_X_maxes = strided_X_maxes.reshape(strided_X.shape)
        # print(strided_X_maxes)

        dL_dX = np.zeros(self.cache['X_shape'])
        
        for i in range(H_out):
            for j in range(W_out):
                dL_dX[:,:,i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size] += strided_X_maxes[:,:,i,j]
        
        return dL_dX


    def backward(self, dL_dy):
        if self.same_kernel_stride:
            return self.fully_vectorized_backward(dL_dy)
        else:
            return self.partially_vectorized_backward(dL_dy)


###################################################################################################################
###################################################################################################################
###################################################################################################################

# model/nn/Flatten.py

class Flatten(Base):
    def __init__(self):
        super().__init__()
        self.name = "Flatten"

    def forward(self, X):
        if self.trainable: self.cache['X_shape'] = X.shape
        output = X.reshape(X.shape[0], -1)
        return output

    def backward(self, dL_dy):
        dL_dX = dL_dy.reshape(self.cache['X_shape'])
        return dL_dX

###################################################################################################################
###################################################################################################################
###################################################################################################################

# model/nn/Linear.py

class Linear(Base):
    def __init__(self, in_features, out_features, lazy_init=False, bias=True):
        super().__init__()
        self.name = "Linear"
        self.params = {
            "in_features": None if lazy_init else in_features,
            "out_features": out_features,
            "bias": bias,
        }
        if not lazy_init: self.state_dict = self.initialize_parameters()
        else: self.state_dict = {"weight": None}
        self.cache = {}

    def initialize_parameters(self):
        # xavier initialization
        # https://paperswithcode.com/method/he-initialization
        std = np.sqrt(2 / self.params["in_features"])
        weights = np.random.randn(self.params["out_features"], self.params["in_features"]) * std
        if self.params["bias"]:
            bias = np.zeros(self.params["out_features"])
            return {"weight": weights, "bias": bias}
        return {"weight": weights}

    def forward(self, X):
        '''
        X shape should be (N, in_features)
        W shape is (out_features, in_features)
        so the output shape is (N, out_features)
        '''
        if self.state_dict["weight"] is None: 
            self.params["in_features"] = X.shape[1]
            self.state_dict = self.initialize_parameters()
        if self.trainable: self.cache['X'] = X
        output = np.dot(X, self.state_dict["weight"].T)
        if self.params["bias"]: output += self.state_dict["bias"]
        return output

    def backward(self, dL_dy):
        '''
        dL_dy = gradient of the cost with respect to the output of the linear layer -> (bs, out_features)
        '''
        # gradient of the cost with respect to the weights
        dL_dW = np.dot(dL_dy.T, self.cache['X'])  # (out_features, bs) * (bs, in_features) -> (out_features, in_features)
        # gradient of the cost with respect to the input
        dL_dX = np.dot(dL_dy, self.state_dict["weight"]) # (bs, out_features) * (out_features, in_features) -> (bs, in_features)
        # gradient of the cost with respect to the bias
        if self.params["bias"]: dL_db = np.sum(dL_dy, axis=0)

        # update weights and bias
        self.grads = {"weight": dL_dW} 
        if self.params["bias"]: self.grads["bias"] = dL_db
        
        return dL_dX


###################################################################################################################
###################################################################################################################
###################################################################################################################

# model/nn/Activations.py

class ReLU(Base):
    def __init__(self):
        super().__init__()
        self.name = 'ReLU'

    def forward(self, X):
        self.cache['X_relu_IDX'] = X>0
        return X*self.cache['X_relu_IDX']

    def backward(self, dL_dy):
        return dL_dy*self.cache['X_relu_IDX']


class Softmax(Base):
    def __init__(self):
        super().__init__()
        self.name = "Softmax"

    def forward(self, X):
        '''
        X is a 2D array of shape (N, #classes)
        '''
        # https://stackoverflow.com/questions/54880369/implementation-of-softmax-function-returns-nan-for-high-inputs
        exp = np.exp(X - np.max(X))
        return exp / np.sum(exp, axis=1, keepdims=True)   

    def backward(self, dL_dy):
        '''
        Here derivation has come from cross_entroy; already has the softmax considered
        '''
        return dL_dy

###################################################################################################################
###################################################################################################################
###################################################################################################################

# model/Model

class Model:
    def __init__(self, config=False, model_layers=[]):
        self.config = config['model'] if config else False
        if self.config: self.layers = self.create_model()
        else: self.layers = model_layers

        if config:
            print("Testing model shapes with random X: ")
            X = np.random.randn(1, 3, config['augment']['img_shape'][0], config['augment']['img_shape'][1])
            self.forward(X, debug=True)
            print('#'*50)

    def forward(self, X, debug=False):
        if debug: print("Input X: -> \t\t", X.shape)
        for i, layer in enumerate(self.layers):
            X = layer(X)
            if debug: print(f"Layer {i}: {layer.name} ->\t", X.shape)
        return X

    def backward(self, dL_dy):
        # print("INPUT BACKWARD: ",dL_dy.shape)
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy)
        return dL_dy

    def __call__(self, X):
        return self.forward(X)

    def __str__(self):
        print_data = "MODEL LAYERS & PARAMETERS: \n"
        for i, layer in enumerate(self.layers):
            print_data += f"Layer {i}: " + str(layer) + "\n"
        return print_data

    def create_model(self):
        model = []

        for layer in self.config:
            name = layer[0]
            params = layer[1]

            if name == "Conv2D":
                model.append(Conv2D(*params))
            elif name == "MaxPool2D":
                model.append(MaxPool2D(*params))
            elif name == "Flatten":
                model.append(Flatten())
            elif name == "Linear":
                model.append(Linear(None, params[0], lazy_init=True))
            elif name == "ReLU":
                model.append(ReLU())
            elif name == "Softmax":
                model.append(Softmax())
        return model


    def save_model(self, path, epoch, wandb_id, cur_lr):
        params = []
        for layer in self.layers:
            params.append(layer.state_dict)

        save_data = {
            'state_dict': params,
            'epoch': epoch,
            'wandb_id': wandb_id,
            'lr': cur_lr,
        }

        with open(path, "wb") as f:
            pickle.dump(save_data, f)

    def load_model(self, path, pretrained=False):
        with open(path, "rb") as f:
            params = pickle.load(f)
            if isinstance(params, dict): 
                print("Loading Epochs: ", params['epoch'], " LR: ", params['lr'], " W&B ID: ", params['wandb_id'])
                params = params['state_dict']
        if pretrained:
            if params[0]['kernels'].shape != self.layers[0].state_dict['kernels'].shape:
                print("Pretrained model input shape does not match model shape")
                # pretrained on grayscale images and train on rgb ones
                b = np.zeros(self.layers[0].state_dict['kernels'].shape)
                b[:, 0, :, :] = params[0]['kernels'][:, 0, :, :]
                b[:, 1, :, :] = params[0]['kernels'][:, 0, :, :]
                b[:, 2, :, :] = params[0]['kernels'][:, 0, :, :]
                print(b.shape, params[0]['kernels'].shape)
                params[0]['kernels'] = b
        for i, layer in enumerate(self.layers):
            layer.state_dict = params[i]
        print("Successfully loaded from " + path)

    # makes model trainable -> stores cache
    def train(self):
        for layer in self.layers:
            layer.trainable = True

    # makes model untrainable -> doesnt store cache
    def eval(self):
        for layer in self.layers:
            layer.trainable = False
        

###################################################################################################################
###################################################################################################################
###################################################################################################################

# model/Loss
class CrossEntropyLoss():
    def __init__(self):
        self.name = "CrossEntropy"

    def __call__(self, y_pred, y_true):
        # cross entropy loss for y_pred of shape (N, #classes)
        # y_true is a one-hot encoded vector of shape (N, #classes)
        return np.sum(-np.sum(y_true * np.log(y_pred), axis=1)) / y_pred.shape[0]

    def get_grad_wrt_softmax(self, y_pred, y_true):
        grad = y_pred - y_true
        return grad/y_pred.shape[0]

###################################################################################################################
###################################################################################################################
###################################################################################################################

# model/LRScheduler
class ReduceLROnPlateau:
    def __init__(self, factor=0.1, patience=10, verbose=0):
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.best_metric = 0
        self.counter = 0

    def step(self, metric, optimizer):
        if metric > self.best_metric:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                optimizer.lr *= self.factor
                self.counter = 0
                print(f"Reduced learning rate to {optimizer.lr}")
                            


###################################################################################################################
###################################################################################################################
###################################################################################################################

# model/Metrics.py
def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def macro_f1(y_true, y_pred):
    # Macro -> Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    return f1_score(y_true, y_pred, average='macro')

def confusion_matrix_sk(y_true, y_pred, labels=None):
    return confusion_matrix(y_true, y_pred)

###################################################################################################################
###################################################################################################################
###################################################################################################################

# model/Optimizer.py
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, model):
        for layer in model.layers:
            layer.update_weights(self.lr)


###################################################################################################################
###################################################################################################################
###################################################################################################################


# utils.py

def get_near_duplicate_removed_train():
    # https://www.kaggle.com/code/nexh98/cse-472-offline4-make-dataset/data?scriptVersionId=118126349
    # This is the dataset with near duplicate images removed using CNN embeddings
    df1 = pd.read_csv("nd_removed_train_a.csv")
    df3 = pd.read_csv("nd_removed_train_c.csv")

    df1 = df1[df1.included == True].reset_index(drop=True)
    df3 = df3[df3.included == True].reset_index(drop=True)
    return df1, df3

def split_dataset(parent_dir="NumtaDB_with_aug", validation_percentage=0.2):
    df1 = pd.read_csv(f"{parent_dir}/training-a.csv")
    df2 = pd.read_csv(f"{parent_dir}/training-b.csv")
    df3 = pd.read_csv(f"{parent_dir}/training-c.csv")

    # df1, df3 = get_near_duplicate_removed_train() 
    df = pd.concat([df1, df2, df3], ignore_index=True)
    df['img_path'] = df['database name'] + '/' + df['filename']

    df['split_col'] = df['database name original'] + '_' + df['digit'].astype(str)
    df = df.sample(frac=1) # shuffle

    split_col = df['split_col'].unique().tolist()
    train_indexes = []
    for cat in split_col:
        indexes = df[df['split_col'] == cat].index.tolist()
        train_indexes.extend(indexes[:int(len(indexes) * (1 - validation_percentage))])
    train_df = df.loc[train_indexes]
    val_df = df.drop(train_indexes)
    print("Train: ", train_df.shape, "; Valid: ", val_df.shape)

    # save csv
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    return train_df, val_df


def one_hot_encoding(y, num_class):
    bs = y.shape[0]
    label = np.zeros((bs, num_class))
    label[np.arange(bs), y] = 1
    return label


def mixup(images, labels, num_classes=10):
    changed_indices = np.random.permutation(images.shape[0])
    # alpha beta values from https://github.com/ultralytics/yolov5/issues/3380
    lam = np.random.beta(8.0, 8.0) 

    changed_images = images[changed_indices]
    changed_labels = labels[changed_indices]

    labels = one_hot_encoding(labels, num_classes)
    changed_labels = one_hot_encoding(changed_labels, num_classes)

    images = lam * images + (1 - lam) * changed_images
    labels = lam * labels + (1 - lam) * changed_labels
    return images, labels


def set_seed(seed):
    np.random.seed(seed)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name):
        self.name = name
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def update_loggings(loggings, epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1):
    loggings['epoch'].append(epoch)
    loggings['train_loss'].append(train_loss)
    loggings['train_acc'].append(train_acc)
    loggings['train_f1'].append(train_f1)
    loggings['val_loss'].append(val_loss)
    loggings['val_acc'].append(val_acc)
    loggings['val_f1'].append(val_f1)
    return loggings


def visualize_training(loggings, save_dir):
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    ax[0].plot(loggings['epoch'], loggings['train_loss'], label='train')
    ax[0].plot(loggings['epoch'], loggings['val_loss'], label='val')
    ax[0].set_title('Loss')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(loggings['epoch'], loggings['train_acc'], label='train')
    ax[1].plot(loggings['epoch'], loggings['val_acc'], label='val')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(loggings['epoch'], loggings['train_f1'], label='train')
    ax[2].plot(loggings['epoch'], loggings['val_f1'], label='val')
    ax[2].set_title('F1 Score')
    ax[2].legend()
    ax[2].grid()

    plt.legend()

    plt.savefig(f'{save_dir}/metrics.png', bbox_inches='tight')
    # plt.show()

# wandb stuff
def wandb_init(config):
    if config['wandb']['entity'] == 'anonymous':
        print("Anonymouse run wandb")
        wandb.login(anonymous="must", relogin=True)
        run = wandb.init(anonymous="allow")
    else:
        if config['resume']:
            with open(config['checkpoint_path'], "rb") as f:
                wandb_id = pickle.load(f)['wandb_id']

        run = wandb.init(
            project=config['wandb']['project'], 
            entity=config['wandb']['entity'],
            name=config['name'],
            config=config, 
            resume="allow",
            id=wandb_id if config['resume'] else None
        )
    return run


def update_wandb(epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, lr):
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_f1': train_f1,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1,
        'lr': lr
    })

###################################################################################################################
###################################################################################################################
###################################################################################################################


def train_one_epoch(model, train_loader, config, optimizer):
    celoss = CrossEntropyLoss()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    f1_meter = AverageMeter('f1')

    model.train()
    for step, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        images, labels = data[0], data[1]

        if np.random.rand() < config['augment']['mixup']:
            images, one_hot_labels = mixup(images, labels, config['num_class'])
        else:
            one_hot_labels = one_hot_encoding(labels, config['num_class'])
        
        out = model(images)
        loss = celoss(out, one_hot_labels)
        model.backward(celoss.get_grad_wrt_softmax(out, one_hot_labels))
        optimizer.step(model) # optimizer step

        out = np.argmax(out, axis=1)
        acc = accuracy(labels, out)
        macf1 = macro_f1(labels, out)

        loss_meter.update(loss)
        acc_meter.update(acc)
        f1_meter.update(macf1)

    return model, loss_meter.avg, acc_meter.avg, f1_meter.avg

def validate_one_epoch(model, val_loader, config):
    celoss = CrossEntropyLoss()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    f1_meter = AverageMeter('f1')

    model.eval()
    for step, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        images, labels = data[0], data[1]
        one_hot_labels = one_hot_encoding(labels, config['num_class'])
        
        out = model(images)
        loss = celoss(out, one_hot_labels)

        out = np.argmax(out, axis=1)
        acc = accuracy(labels, out)
        macf1 = macro_f1(labels, out)

        loss_meter.update(loss)
        acc_meter.update(acc)
        f1_meter.update(macf1)

    return loss_meter.avg, acc_meter.avg, f1_meter.avg


def fit_model(model, train_loader, val_loader, config, wandb_run):
    # save based on macro f1
    best_macro_f1 = 0

    loggings = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    
    if config['resume']:
        with open(config['checkpoint_path'], "rb") as f:
            prev_run = pickle.load(f)
            epoch = prev_run['epoch']
            lr = prev_run['lr']

    optimizer = SGD(lr=lr if config['resume'] else config['lr'])
    scheduler = ReduceLROnPlateau(factor=config['lr_scheduler']['factor'], patience=config['lr_scheduler']['patience'], verbose=1)


    start_epoch = epoch if config['resume'] else 0
    for epoch in range(start_epoch, config['epochs']):
        model, train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, config, optimizer)
        val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, config)

        print(f"Epoch {epoch+1}/{config['epochs']} => LR {optimizer.lr}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Macro F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

        scheduler.step(val_f1, optimizer) # reduce lr based on validation f1 performance

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            model.save_model(f"{config['output_dir']}/best_model_E{epoch}.npy", epoch, config['wandb_id'], optimizer.lr)

        loggings = update_loggings(loggings, epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1)
        if config['use_wandb']: update_wandb(epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, optimizer.lr)
    
    loggings = pd.DataFrame(loggings)
    loggings.to_csv(f"{config['output_dir']}/logs.csv", index=False)
    
    if config['use_wandb']:
        wandb_run.summary[f"Best VAL MacroF1"] = best_macro_f1
        wandb_run.summary[f"Best VAL Accuracy"] = loggings['val_acc'].max()
        wandb_run.summary[f"Best VAL Loss"] = loggings['val_loss'].min()
        wandb_run.finish()

    visualize_training(loggings, config['output_dir'])


###################################################################################################################
###################################################################################################################
###################################################################################################################

# train.py

if __name__ == "__main__":
    set_seed(42)

    # read config
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    os.makedirs(config['output_dir'], exist_ok=True)
    wandb_run = wandb_init(config) if config['use_wandb'] else None
    config['wandb_id'] = wandb_run.id if config['use_wandb'] else None
    
    # make model
    model = Model(config)
    print(model)
    if config['checkpoint_path']:
        model.load_model(config['checkpoint_path'], pretrained=True)

    # make dataset
    train_df, valid_df = split_dataset(config['data_dir'], validation_percentage=0.2)
    print("Train: ", train_df.shape, "; Valid: ", valid_df.shape)
    
    if config['debug']:
        train_df = train_df.sample(frac=1).reset_index(drop=True)[:100]
        valid_df = valid_df.sample(frac=1).reset_index(drop=True)[:100]

    train_dataset = Dataset(config['data_dir'], train_df, label_col='digit', mode='train', config=config['augment'])
    valid_dataset = Dataset(config['data_dir'], valid_df, label_col='digit', mode='valid', config=config['augment'])
    check_dataset(train_dataset, valid_dataset, save_dir=config['output_dir'])

    train_loader = DataLoader(train_dataset, batch_size=config['train_batch'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['valid_batch'], shuffle=False)
    if config['augment']['mixup']:
        data = next(iter(train_loader))
        images, labels = data[0], data[1]
        images, labels = mixup(images, labels, num_classes=config['num_class'])

        check_dataset([images, labels], valid_dataset, save_dir=config['output_dir'], from_mixup=True)

    # train
    fit_model(model, train_loader, valid_loader, config, wandb_run)
