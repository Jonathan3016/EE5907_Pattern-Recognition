import random
import numpy as np
import pandas as pd
from model.PCA import PCA
from model.KNN import kNN

# Read from training and testing set
train_x = np.load('../data/train_x.npy')
train_y = np.load('../data/train_y.npy')
test_x = np.load('../data/test_x.npy')
test_y = np.load('../data/test_y.npy')

def run_model_pca():

    """
        Randomly sample 500 images from the CMU PIE training set and your own photos.
    """

    # randomly select 493 samples from CMU PIE and 7 from selfies
    cmupie_idx = random.sample([idx for idx in range(len(train_x) - 7)], 493)
    selfie_idx = [idx for idx in range(len(train_x) - 7, len(train_x))]
    pca_dataset = np.concatenate([train_x[cmupie_idx], train_x[selfie_idx]], 0)
    pca_label = np.concatenate([train_y[cmupie_idx], ['selfie' for idx in range(7)]], 0)

    """
        Apply PCA to reduce the dimensionality of vectorized images to 2 and 3 respectively.
        Visualize the projected data vector in 2d and 3d plots.
        Highlight the projected points corresponding to your photo.
        Also visualize the corresponding 3 eigenfaces used for the dimensionality reduction.
    """

    # import PCA and using the functions in model PCA
    pca = PCA(pca_dataset)
    # Reduce the dimension to 2 and 3
    x_2d = pca.reduce_train_dim(n_component=2)
    pca.plot_distribution(x_2d)
    x_3d = pca.reduce_train_dim(n_component=3)
    pca.plot_distribution(x_3d)
    # Plot eigenfaces
    pca.plot_eigenfaces()

    """
         Apply PCA to reduce the dimensionality of face images to 40, 80 and 200 respectively.
         Classifying the test images using the rule of nearest neighbor.
         Report the classification accuracy on the CMU PIE test images and your own photo separately.
    """

    # Classification with PCA, KNN
    dimensions = np.array([40, 80, 200])  # Set the dimensions to 40, 80 and 200
    k_near = 1  # Default the value of k-nearest as k=1
    pca_result = pd.DataFrame(columns=['Dim', 'acc@CMU PIE', 'acc@Selfie', 'acc@Total'])
    for idx, dim in enumerate(dimensions):
        train_reduce = pca.reduce_train_dim(n_component=dim)
        test_reduce = pca.reduce_test_dim(test_x)
        acc_cmupie, acc_selfie, acc_total = kNN(train_reduce, pca_label, test_reduce, test_y, k_near)
        pca_result.loc[idx, 'Dim'] = dim
        pca_result.loc[idx, 'acc@CMU PIE'] = acc_cmupie
        pca_result.loc[idx, 'acc@Selfie'] = acc_selfie
        pca_result.loc[idx, 'acc@Total'] = acc_total
    pca_result.to_csv('../result/pca/pca_accuracy.csv', index=False)


if __name__ == '__main__':
    run_model_pca()
