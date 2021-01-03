import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

## HYPER PARAMETERS - note these can be overwritten in script
# Various parameters
#seq_cutoff_speed = 45
#seq_cutoff_time = 60
filter_seq = 5 #trade-off: might yield multiple classes in one sequence
#label = 'labelM' 
label = 'label2' 
#label = 'label2'

# Network Parameters
IMG_INPUT_H_W = 9, 9
IMG_INPUT_C = 11
TOD_INPUT_DIM = 6 # why ? 
CNN_HIDDEN_DIM = 16
CNN_HIDDEN_DIM_2 = 8
CNN_KERNEL_SIZE = 3
CNN_PADDING = 1

SEQ_LENGTH = filter_seq + 1
SEQ_FEATURES = 2
TOD_INPUT_DIM = 5
RNN_HIDDEN_DIM = 4
RNN_NUM_LAYERS = 2
RNN_BIDIRECT = True
FC_HIDDEN_DIM = 512
DROPOUT_PROP = 0.45

#if label == 'label2':
NUM_CLASSES = 2
#elif label == 'labelM':
#  NUM_CLASSES = 6
#elif label == 'labelC':
#  NUM_CLASSES = 16

# Training Parameters
NUM_EPOCH = 50
BATCH_SIZE = 512
LEARNING_RATE = 0.01
LEARNING_DECAY_FACTOR = 10
LEARNING_DECAY_EPOCHS = [10]
## end of parameters


def adjust_lr(optimizer, epoch):
    number_decay_points_passed = np.sum(epoch >= np.array(LEARNING_DECAY_EPOCHS))
    lr = LEARNING_RATE * (LEARNING_DECAY_FACTOR ** number_decay_points_passed)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

def conv2d_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * padding) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * padding) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

class CnnNet(nn.Module):

    def __init__(self):
        super(CnnNet, self).__init__()
        
        self.cnn_layer_1 = nn.Sequential(
            nn.Conv2d(IMG_INPUT_C, CNN_HIDDEN_DIM, kernel_size=CNN_KERNEL_SIZE, padding=CNN_PADDING),
            nn.BatchNorm2d(CNN_HIDDEN_DIM),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(DROPOUT_PROP)
        )
        
        h, w = IMG_INPUT_H_W
        #print(h, w)
        h, w = conv2d_output_shape((h, w), CNN_KERNEL_SIZE, padding=CNN_PADDING)
        #print(h, w)
        h, w = conv2d_output_shape((h, w), kernel_size=3, stride=2, padding=0)        
        #print(h, w)
        self.cnn_layer_2 = nn.Sequential(
            nn.Conv2d(CNN_HIDDEN_DIM, CNN_HIDDEN_DIM_2, kernel_size=CNN_KERNEL_SIZE, padding=CNN_PADDING),
            nn.BatchNorm2d(CNN_HIDDEN_DIM_2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(DROPOUT_PROP)
        )
               
        h, w = conv2d_output_shape((h, w), CNN_KERNEL_SIZE, padding=CNN_PADDING)
        #print(h, w)
        h, w = conv2d_output_shape((h, w), kernel_size=2, stride=2, padding=0)
        #print(h, w)
        
        rnn_output_dim = SEQ_LENGTH*RNN_HIDDEN_DIM
        if RNN_BIDIRECT:
            rnn_output_dim *= 2
        
        self.rnn_layer = nn.Sequential(
            nn.GRU(SEQ_FEATURES, RNN_HIDDEN_DIM, RNN_NUM_LAYERS, batch_first=True, bidirectional=RNN_BIDIRECT),
            SelectItem(0),
            nn.Flatten(start_dim=1),
            nn.BatchNorm1d(rnn_output_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_PROP)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(CNN_HIDDEN_DIM_2 * h * w + TOD_INPUT_DIM + rnn_output_dim, FC_HIDDEN_DIM),
            nn.BatchNorm1d(FC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_PROP),
            nn.Linear(FC_HIDDEN_DIM, FC_HIDDEN_DIM//5),
            nn.BatchNorm1d(FC_HIDDEN_DIM//5),
            nn.ReLU(),
            nn.Dropout(DROPOUT_PROP),
            nn.Linear(FC_HIDDEN_DIM//5, NUM_CLASSES),
            #nn.BatchNorm1d(NUM_CLASSES),
            #nn.ReLU(),
            #nn.Dropout(DROPOUT_PROP)
        )
        #print(CNN_HIDDEN_DIM_2 * h * w + TOD_INPUT_DIM + rnn_output_dim)      

    def forward(self, X_img, X_seq, X_tod):
        X_img = X_img.permute(0, 3, 1, 2)
        #print(f'1     {X_seq.shape}')
        out_img = self.cnn_layer_1(X_img)
        #print(f'2     {X_img.shape}')
        out_img = self.cnn_layer_2(out_img)
        #print(f'3     {out_img.shape}')
        out_img = out_img.reshape(out_img.size(0), -1)
        #print(f'4     {out_img.shape}')
        out_seq = self.rnn_layer(X_seq)
        #print(f'5     {X_seq.shape}')
        out = torch.cat([out_seq,out_img, X_tod], dim=1)
        #print(f'6     {out.shape}')
        out = self.fc_layer(out)
        
        # Softmax is implemented within PyTorch the cross-entropy
        return out
  
class ImageTensorDataset(torch.utils.data.Dataset):

    def __init__(self, df, image_data, filter_seq=filter_seq, label=label):
        # consider sequences of length 5
        self.seq = np.stack([np.roll(df[['delta_d', 'bearing']].values, i, axis = 0) for i in range(filter_seq, -1, -1)], axis = 1) 
        self.seq = self.seq[df['segment_ix'] >= filter_seq]
        
        self.labels = df[df['segment_ix'] >= filter_seq][label].values        
        self.user_id = df[df['segment_ix'] >= filter_seq]['user'].values
        self.image_ix = df[df['segment_ix'] >= filter_seq]['image_ix'].values        
        self.image_data = image_data # why?
        tod = df[df['segment_ix'] >= filter_seq]['tod'].values # what ? 
        self.tod_one_hot = np.eye(5)[tod]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, key):
        image = self.image_data[self.user_id[key]][self.image_ix[key]]
        return image, self.seq[key], self.tod_one_hot[key], self.labels[key]