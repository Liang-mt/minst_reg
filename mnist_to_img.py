import os
import torch
import torchvision
import torchvision.transforms as transforms
from skimage import io
import torchvision.datasets.mnist as mnist
import numpy

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# MNIST dataset
train_dataset = torchvision.datasets.MNIST("C:/Users/28645/Desktop/minst_reg/",train=True,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST("C:/Users/28645/Desktop/minst_reg/",train=False,transform=transforms.ToTensor(),download=True)

root = './MNIST/raw/'

train_set = (mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
             mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
             )
test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
)


def convert_to_img(train=True):
    if (train):
        data_path = './data/data_train01/'

        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
            img_path = data_path + str(label.item()) + '/'
            print('train_img_path:', img_path, 'img_num:', i)
            img_name = img_path + str(i) + '.png'
            if (not os.path.exists(img_path)):
                os.makedirs(img_path)
            io.imsave(img_name, img.numpy())
    else:
        data_path = './data/data_test01/'

        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            img_path = data_path + str(label.item()) + '/'
            print('test_img_path:', img_path, 'img_numpy:', i)
            img_name = img_path + str(i) + '.png'
            if (not os.path.exists(img_path)):
                os.makedirs(img_path)
            io.imsave(img_name, img.numpy())


convert_to_img(True)
convert_to_img(False)
