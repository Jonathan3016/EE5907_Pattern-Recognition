import numpy as np
import pandas as pd
from model.PCA import PCA
from model.SVM import SVM

# Read from training and testing set
train_x = np.load('../data/train_x.npy')
train_y = np.load('../data/train_y.npy')
test_x = np.load('../data/test_x.npy')
test_y = np.load('../data/test_y.npy')

# SVM requires that the label cannot use strings
train_y[train_y == 'selfie'] = 70
test_y[test_y == 'selfie'] = 70
test_y = test_y.astype(np.float64)


def run_model_svm():

    """
    Use the raw face images (vectorized) and the face vectors after PCA pre-processing
    (with dimensionality of 80 and 200) as inputs to linear SVM.
    Try values of the penalty parameter C in {0.01, 0.1, 1}.
    Report the classification accuracy with different parameters and dimensions.
    """

    dimensions = np.array(['raw', 80, 200])
    c_list = np.array([0.01, 0.1, 1])  # Penalty parameter C in {0.01, 0.1, 1}
    svm_result = pd.DataFrame(columns=['Dim', 'C', 'Accuracy(%)'])

    for idx1, dim in enumerate(dimensions):

        # Use the raw face images (vectorized)
        if dim == 'raw':
            train_svm = train_x.reshape(-1, 32*32)
            test_svm = test_x.reshape(-1, 32*32)

        # Use the face vectors after PCA pre-processing
        else:
            pca = PCA(train_x)
            train_svm = pca.reduce_train_dim(n_component=int(dim))
            test_svm = pca.reduce_test_dim(test_x)

        # Classification using SVM
        for idx2, c in enumerate(c_list):
            idx = idx1 * 3 + idx2
            svm = SVM(c)
            pred_label, pred_acc, pred_val = svm.svm_fit_pred(train_svm, train_y, test_svm, test_y)
            svm_result.loc[idx, 'Dim'] = dim
            svm_result.loc[idx, 'C'] = c
            svm_result.loc[idx, 'Accuracy(%)'] = pred_acc[0]

    svm_result.to_csv('./svm_accuracy.csv', index=False)


if __name__ == '__main__':
    run_model_svm()
