import tensorflow as tf
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from utils import *


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_dataset(dataset):
    images, labels = dataset[0]
    print("IMAGES: ", images.shape)
    print("LABELS: ", labels.shape)
    print(onehot(labels))
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



def get_cnn_dataset(dataset, batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    if dataset == "cifar":
        trainset = torchvision.datasets.CIFAR10(root='./cifar_data', train=True,
                                                download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
        train_data = list(iter(trainloader))
        testset = torchvision.datasets.CIFAR10(root='./cifar_data', train=False,
                                               download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True)
        test_data = list(iter(testloader))
    elif dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True,
                                                download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
        train_data = list(iter(trainloader))
        testset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=False,
                                               download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True)
        test_data = list(iter(testloader))
    elif dataset == "svhn":
        trainset = torchvision.datasets.SVHN(root='./svhn_data', split='train',
                                                download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
        train_data = list(iter(trainloader))
        testset = torchvision.datasets.SVHN(root='./svhn_data', split='test',
                                               download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True)
        test_data = list(iter(testloader))
    elif dataset == "mnist":
        mnist_transform = transforms.Compose([transforms.ToTensor(), mnist_normalize])
        trainset = torchvision.datasets.MNIST(root='./mnist_data', train=True,
                                                download=False, transform=mnist_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
        train_data = list(iter(trainloader))
        testset = torchvision.datasets.MNIST(root='./mnist_data', train=False,
                                               download=False, transform=mnist_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True)
        test_data = list(iter(testloader))
    else:
        raise Exception("dataset: " + str(dataset) + " not supported")

    print("Setup data:")
    print("Train: ",len(train_data))
    print("Test: ", len(test_data))
    return train_data, test_data


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def get_lstm_dataset(seq_length, batch_size,buffer_size=10000):
    path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])
    examples_per_epoch = len(text)//(seq_length+1)

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    dataset = sequences.map(split_input_target)

    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    dataset = list(iter(dataset))
    #get dataset in right format
    vocab_size = len(vocab)
    return dataset, vocab_size,char2idx,idx2char

