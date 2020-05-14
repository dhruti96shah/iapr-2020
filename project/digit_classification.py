import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def pred_digit(img_to_pred, network):
	with torch.no_grad():
		output = network(img_to_pred)
	return output.data.max(1, keepdim=True)[1][0].item(), output.data.max(1, keepdim=True)[0][0].item()

if __name__ == '__main__':
	test_loader = torch.utils.data.DataLoader(
	  torchvision.datasets.MNIST('./data/', train=False, download=True,
	                             transform=torchvision.transforms.Compose([
	                               torchvision.transforms.ToTensor(),
	                               torchvision.transforms.Normalize(
	                                 (0.1307,), (0.3081,))
	                             ])),
	  batch_size=1000, shuffle=True)
	examples = enumerate(test_loader)
	batch_idx, (example_data, example_targets) = next(examples)
	print(example_targets[0])
	print(pred_digit(example_data[0].unsqueeze(0)))
