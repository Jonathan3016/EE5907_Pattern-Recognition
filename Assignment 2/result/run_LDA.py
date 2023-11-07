import random
import numpy as np
import pandas as pd
from model.LDA import LDA
from model.KNN import kNN

# Read from training and testing set
train_x = np.load('../data/train_x.npy')
train_y = np.load('../data/train_y.npy')
test_x = np.load('../data/test_x.npy')
test_y = np.load('../data/test_y.npy')

def run_model_lda(mode):

    # randomly select 493 samples from CMU PIE and 7 from selfies
    if mode == '500_samples':
        cmupie_idx = random.sample([idx for idx in range(len(train_x) - 7)], 493)
        selfie_idx = [idx for idx in range(len(train_x) - 7, len(train_x))]
        lda_dataset = np.concatenate([train_x[cmupie_idx], train_x[selfie_idx]], 0)
        lda_label = np.concatenate([train_y[cmupie_idx], ['selfie' for idx in range(7)]], 0)

    # Use all samples in training set
    else:
        lda_dataset = train_x
        lda_label = train_y

    """
        Apply LDA to reduce data dimensionality from to 2, 3 and 9.
        Visualize distribution of the sampled data with dimensionality of 2 and 3 respectively.
        Report the classification accuracy for data with dimensions of 2, 3 and 9 respectively.
        Based on nearest neighbor classifier.
        Report the classification accuracy on the CMU PIE test images and your own photo separately.
    """
    # import LDA and using the functions in model LDA
    lda = LDA(lda_dataset, lda_label)

    # Classification with LDA, KNN
    dimensions = np.array([2, 3, 9])  # Set the dimensions to 2, 3 and 9
    k_near = 1  # Default the value of k-nearest as k=1
    lda_result = pd.DataFrame(columns=['Dim', 'acc@CMU PIE', 'acc@Selfie', 'acc@Total'])
    if mode == '500_samples':
        title = './lda_500_samples_accuracy.csv'
    else:
        title = './lda_all_samples_accuracy.csv'
    for idx, dim in enumerate(dimensions):
        train_reduced = lda.reduce_train_dim(n_component=dim)
        if mode == '500_samples' and dim != 9:
            lda.plot_distribution(train_reduced)
        test_reduced = lda.reduce_test_dim(test_x)
        acc_cmupie, acc_selfie, acc_total = kNN(train_reduced, lda_label, test_reduced, test_y, k_near)
        lda_result.loc[idx, 'Dim'] = dim
        lda_result.loc[idx, 'acc@CMU PIE'] = acc_cmupie
        lda_result.loc[idx, 'acc@Selfie'] = acc_selfie
        lda_result.loc[idx, 'acc@Total'] = acc_total
    lda_result.to_csv(title, index=False)


if __name__ == '__main__':
    run_model_lda(mode='500_samples')
    # run_model_lda(mode='all_samples')
