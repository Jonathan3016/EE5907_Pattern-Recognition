import os
import glob
import numpy as np
import matplotlib.image as mpimg
import cv2

num_PIE = 68
num_selfie = 10
num_seed = 25
random_seed = 46

def generate_dataset():

    """
    Choose the 25 subjects randomly from the CMU PIE dataset and 10 selfies
    For each subject, use 70% for training and the remaining 30% for testing
    """

    # Randomly selected subjects
    np.random.seed(random_seed)
    random_idx = np.sort(np.random.choice(range(num_PIE), num_seed, replace=False))
    print('Randomly selected subjects for this project:', random_idx)

    # Adjusting selfie images
    selfie_path = '../Selfie/src'
    selfie_list = os.listdir(selfie_path)
    for i, img in enumerate(selfie_list):
        img_path = os.path.join(selfie_path, img)
        selfie = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        selfie = cv2.resize(selfie, (32, 32))
        cv2.imwrite(os.path.join('../Selfie/imgs', '{}.jpg'.format(i)), selfie)

    train_x, train_y = [], []
    test_x, test_y = [], []

    # Read from CMU PIE dataset
    for idx in range(num_seed):
        sub_idx = random_idx[idx]
        path_samples = glob.glob('../PIE/{}/*.jpg'.format(sub_idx))
        n_train_samples = round(len(path_samples) * 0.7)
        # Training set
        for img in path_samples[:n_train_samples]:
            sample = mpimg.imread(img)
            train_x.append(sample)
            train_y.append(sub_idx)
        # Testing set
        for img in path_samples[n_train_samples:]:
            sample = mpimg.imread(img)
            test_x.append(sample)
            test_y.append(sub_idx)

    # Read from selfies
    n_train_samples = round(num_selfie * 0.7)
    # Training set
    for idx in range(n_train_samples):
        sample = mpimg.imread('../Selfie/imgs/{}.jpg'.format(idx))
        if len(sample.shape) == 3:
            sample = sample[..., 0]
        train_x.append(sample)
        train_y.append('selfie')
    # Testing set
    for idx in range(n_train_samples, num_selfie):
        sample = mpimg.imread('../Selfie/imgs/{}.jpg'.format(idx))
        if len(sample.shape) == 3:
            sample = sample[..., 0]
        test_x.append(sample)
        test_y.append('selfie')

    train_x = np.stack(train_x, 0)
    test_x = np.stack(test_x, 0)
    np.save('./train_x.npy', train_x)
    np.save('./train_y.npy', train_y)
    np.save('./test_x.npy', test_x)
    np.save('./test_y.npy', test_y)

    print('The shape of training set: {}, the shape of testing set: {}'.format(train_x.shape, test_x.shape))


if __name__ == '__main__':
    # Generate dataset
    generate_dataset()