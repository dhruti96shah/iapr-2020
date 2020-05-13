import torch
import torchvision
from torchvision import transforms
import argparse
from localization import *
from operator_classification import *
import numpy as np
from PIL import Image
import scipy.misc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(92416, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
def train(model,epoch,train_loader):
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), './model_data/model.pth')
            torch.save(optimizer.state_dict(), './model_data/optimizer.pth')
    
def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    
    
def load_mnist(path,size):
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(path, train=True, download=True,
                             transform=torchvision.transforms.Compose([
                                 torchvision.transforms.Resize(size),
                                 torchvision.transforms.RandomRotation(degrees=(-90,90),fill=(0,)),
                              torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])), batch_size=batch_size_train, shuffle=True)
               
    test_loader = torch.utils.data.DataLoader( torchvision.datasets.MNIST(path, train=False, download=True,
                             transform=torchvision.transforms.Compose([
                                 torchvision.transforms.Resize(size),
                                 torchvision.transforms.RandomRotation(degrees=(-90,90),fill=(0,)),
                              torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])), batch_size=batch_size_test, shuffle=True)
    return train_loader, test_loader


def predict (model,img,trans):
    img_t = trans(img)
    batch_t = torch.unsqueeze(img_t, 0)    
    output = model(batch_t)
    pred = output.data.max(1, keepdim=True)[1].item()
    return pred

def pred_digit(model,trans,img,training_flag):
    
    assert training_flag== False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img,axis=2) 
    
    # set CNN parameters
    train = False
    n_epochs = 1
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    mnist_path = './data'
    size = 80 
    
    # train if needed
    if training_flag == True:
        # load mnist data for training
        train_loader, test_loader = load_mnist(path,size)
        model = Net()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum)
        for epoch in range(1, n_epochs + 1):
            train(epoch)
            test()
     
    
    
    prediction = predict(model,img,trans)
    
      
    return prediction