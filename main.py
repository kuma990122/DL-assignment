import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data


# The function for uncompressing the batch file
def unpickle(files):
    list = []
    for file in files:
        with open(file,'rb') as f:
            dict = pickle.load(f, encoding='bytes')
            list.append(dict)
    print("Size of List: ", len(list))
    return list

def show_info(data, labels):
    data = np.array(data)
    labels = np.array(labels)

    print('data shape:', data.shape)
    print('data type:', data.dtype)
    print('labels shape:', labels.shape)
    print('labels type:', labels.dtype)


# Defining the function for showing the shape and type of image data and labeling data
def show_image(data, labels, index):
    data = np.array(data)
    labels = np.array(labels)
    image = data[index]
    label = labels[index]
    image = image.reshape(3, 32, 32).transpose(1, 2, 0)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.imshow(image)
    plt.title(classes[label])
    plt.show()

#Create the list of batch files
file1 = 'dataset/data_batch_1'
file2 = 'dataset/data_batch_2'
file3 = 'dataset/data_batch_3'
file4 = 'dataset/data_batch_4'
file5 = 'dataset/data_batch_5'
file6 = 'dataset/test_batch'
files = [file1, file2, file3, file4, file5, file6]

# Create the list of dictionary by using pickle module
dicts = unpickle(files)
print("Size of Dicts: ", len(dicts))
merged_dict = {}
for dict in dicts:
    merged_dict.update(dict)
data = np.array(merged_dict[b'data'])
labels = np.array(merged_dict[b'labels'])
merged_dict['data'] = data
merged_dict['labels'] = labels

data_tensor = torch.from_numpy(data)
labels_tensor = torch.from_numpy(labels)
# Tensorize our dataset
dataset = Data.TensorDataset(data_tensor,labels_tensor)
dataset_size = len(dataset)

print("Size of dataset: ",dataset_size)

# Splitting the dataset into training, validation, testing set
training_size = int(0.7*dataset_size)
validation_size = int(0.15*dataset_size)
testing_size = int(0.15*dataset_size)
train_dataset, val_dataset, test_dataset = Data.random_split(
    dataset, [training_size, validation_size, testing_size])


show_info(data, labels)
for i in range(20):
   show_image(data, labels, i)

# Create dataloader object
train_loader = Data.DataLoader(
dataset=train_dataset,
batch_size=32,
shuffle=True,
num_workers=2
)

val_loader = Data.DataLoader(
dataset=val_dataset,
batch_size=32,
shuffle=True,
num_workers=2
)

test_loader = Data.DataLoader(
dataset=test_dataset,
batch_size=32,
shuffle=True,
num_workers=2
)
