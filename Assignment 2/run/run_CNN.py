import numpy as np
from model.CNN import CNN, Trainer, generate_dataloader
import os


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Read from training and testing set
train_x = np.load('../data/train_x.npy')
train_y = np.load('../data/train_y.npy')
test_x = np.load('../data/test_x.npy')
test_y = np.load('../data/test_y.npy')

train_y[train_y == 'selfie'] = 69
test_y[test_y == 'selfie'] = 69
train_y = train_y.astype(np.int64)
test_y = test_y.astype(np.int64)


class CustomArrayMapper:
    def __init__(self, unique_values):
        self.mapping = {value: i for i, value in enumerate(sorted(unique_values))}

    def __getitem__(self, key):
        return self.mapping.get(key, None)


custom_mapper = CustomArrayMapper(np.unique(train_y))
train_y = np.array([custom_mapper[item] for item in train_y])
custom_mapper = CustomArrayMapper(np.unique(test_y))
test_y = np.array([custom_mapper[item] for item in test_y])

"""
    Requirements：
    Train a CNN with two convolutional layers and one fully connected layer,
    with the architecture specified as follows: number of nodes: 20-50-500-26.
    The number of the nodes in the last layer is fixed as 26 as we are performing 26-category classification.
    Convolutional kernel sizes are set as 5.
    Each convolutional layer is followed by a max pooling layer with a kernel size of 2 and stride of 2.
    The fully connected layer is followed by ReLU.
"""

def run_model_cnn():

    # import CNN and using the functions in model CNN
    cnn = CNN()

    epochs = 500
    batch_size = 32
    train_loader, test_loader = generate_dataloader(train_x, train_y, test_x, test_y, batch_size)

    # import Trainer
    trainer = Trainer(cnn)
    trainer.train_model(epochs, train_loader, test_loader)
    trainer.plot_result()


if __name__ == '__main__':
    run_model_cnn()
