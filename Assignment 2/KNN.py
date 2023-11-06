import numpy as np
from collections import Counter
import sklearn.metrics as metrics

# K-nearest
def kNN(train_x, train_y, test_x, test_y, k):
    preds = []

    # Calculate pairwise distances between training and test points
    dist_matrix = np.linalg.norm(train_x[:, np.newaxis] - test_x, axis=2)

    for i in range(test_x.shape[0]):
        # Find the indices of the k-nearest neighbors
        indices = np.argpartition(dist_matrix[:, i], k)[:k]

        # Extract labels of the k-nearest neighbors
        k_nearest_labels = train_y[indices]

        # Perform majority voting using Counter
        label_counter = Counter(k_nearest_labels)
        most_common_label = label_counter.most_common(1)[0][0]

        preds.append(most_common_label)

    # Get the accuracy of the prediction
    acc_cmupie = metrics.accuracy_score(test_y[:-3], preds[:-3])
    acc_selfie = metrics.accuracy_score(test_y[-3:], preds[-3:])
    acc_total = metrics.accuracy_score(test_y, preds)

    return acc_cmupie, acc_selfie, acc_total
