import pickle
import numpy as np
import matplotlib.pyplot as plt

# The function for uncompressing the batch file
def unpickle(file):
    with open(file,'rb') as f:
        dict = pickle.load(f, encoding= 'bytes' )
    return dict

def show_info(data,labels):
    data = np.array(data)
    labels = np.array(labels)

    print('data shape:', data.shape)
    print('data type:', data.dtype)
    print('labels shape:', labels.shape)
    print('labels type:', labels.dtype)

# Defining the function for showing the shape and type of image data and labeling data
def show_image(data,labels,index):
    data = np.array(data)
    labels = np.array(labels)
    image = data[index]
    label = labels[index]
    image = image.reshape(3,32,32).transpose(1,2,0)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.imshow(image)
    plt.title(classes[label])
    plt.show()

file = 'dataset/data_batch_1'
dict = unpickle(file)
data = dict[b'data']
labels = dict[b'labels']
show_info(data,labels)

for i in range (20):
  show_image(data,labels,i)

