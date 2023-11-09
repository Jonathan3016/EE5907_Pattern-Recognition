import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()
        # Convolutional layer 1: input channels=1, output channels=20, kernel size=5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        # Convolutional layer 2: input channels=20, output channels=50, kernel size=5
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5)
        # Max pooling layer 1&2: kernel size=2, stride=2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer 1: input features=50*5*5, output features=500
        self.fc1 = nn.Linear(in_features=50 * 5 * 5, out_features=500)
        # Fully connected layer 2: input features=500, output features=26
        self.fc2 = nn.Linear(in_features=500, out_features=26)
        # Prevent unstable network performance due to large data size
        self.bn1 = nn.BatchNorm2d(num_features=20)
        self.bn2 = nn.BatchNorm2d(num_features=50)
        self.bn3 = nn.BatchNorm1d(num_features=500)
        self.bn4 = nn.BatchNorm1d(num_features=26)

    def forward(self, x):

        x = self.pool(F.relu(self.bn1(self.conv1(x.unsqueeze(1)))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.contiguous().view(-1, 50 * 5 * 5)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))

        return x


class Trainer:

    def __init__(self, model):

        self.model = model
        self.param = self.model.parameters()
        self.epochs = None
        # Use SGD optimizer for updating model parameters.
        self.optimizer = SGD(self.param, lr=0.005)
        self.loss_func = nn.CrossEntropyLoss()
        # Device configuration - uses GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_epoch = -1
        self.best_acc = 0.0
        self.train_loss = []
        self.test_acc = []

    def train_epoch(self, train_loader):

        self.model.train()  # Set the model to training mode
        train_losses = []

        for idx, [sample, label] in enumerate(train_loader):
            # Move samples and labels to the configured device
            sample, label = sample.to(self.device), label.to(self.device)
            # sample = sample.permute(0, 3, 1, 2)
            self.optimizer.zero_grad()  # Zero the parameter gradients
            output = self.model(sample)  # Forward passS
            loss = self.loss_func(output, label.long())  # Compute the loss
            loss.backward()  # Backward pass
            # Using gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=self.param, max_norm=15, norm_type=2)
            self.optimizer.step()  # Update optimizer
            train_losses.append(loss.item())

        return np.mean(train_losses)

    def test(self, test_loader):

        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for idx, (sample, label) in enumerate(test_loader):
                sample, label = sample.to(self.device), label.to(self.device)
                # sample = sample.permute(0, 3, 1, 2)
                outputs = self.model(sample)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        accuracy = 100 * correct / total

        return accuracy

    def train_model(self, epochs, train_loader, test_loader):

        self.model.to(self.device)  # Move the model to the configured device
        self.epochs = epochs

        for epoch in range(self.epochs):
            average_loss = self.train_epoch(train_loader)
            self.train_loss.append(average_loss)
            test_acc = self.test(test_loader)
            self.test_acc.append(test_acc)
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_epoch = epoch + 1
                self.save_model()
            print(f'Epoch [{epoch + 1}/{self.epochs}] Train Loss: {average_loss:.6f} '
                  f'Test Accuracy: {test_acc:.6f} Best Accuracy: {self.best_acc:.6f}')

        print(f'Best epoch: {self.best_epoch:.6f} with test accuracy: {self.best_acc:.6f}')
        np.save('../result/CNN_train_losses.npy', np.array(self.train_loss))
        np.save('../result/CNN_test_accuracy.npy', np.array(self.test_acc))

    def save_model(self):

        path = '../result/best_model.pth'
        torch.save(self.model.state_dict(), path)

    def plot_result(self):

        plt.figure(figsize=(8, 6))
        plt.plot(range(self.epochs), self.train_loss, label='Train Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title('Train Loss')
        plt.vlines(self.best_epoch, 0, 3, colors="red", linestyles="dashed")
        plt.legend()
        plt.tight_layout()
        plt.savefig('../result/CNN_train_loss.png', bbox_inches='tight')

        plt.figure(figsize=(8, 6))
        plt.plot(range(self.epochs), self.test_acc, label='Test Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.title('Test Accuracy')
        plt.vlines(self.best_epoch, 0, 100, colors="red", linestyles="dashed")
        plt.legend()
        plt.tight_layout()
        plt.savefig('../result/CNN_test_accuracy.png', bbox_inches='tight')


def generate_dataloader(train_x, train_y, test_x, test_y, batch_size):

    train_x = Tensor(train_x)
    # train_y = F.one_hot(Tensor(train_y - 1).to(torch.int64))
    train_y = Tensor(train_y)
    test_x = Tensor(test_x)
    test_y = Tensor(test_y)

    training_set = TensorDataset(train_x, train_y)
    testing_set = TensorDataset(test_x, test_y)
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# cnn = CNN()

# for name, param in cnn.named_parameters():
    # print(name, '-->', param.type(), '-->', param.dtype, '-->', param.shape)
