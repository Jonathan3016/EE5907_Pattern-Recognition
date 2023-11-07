import numpy as np
import matplotlib.pyplot as plt

class LDA:

    def __init__(self, sample, label):

        self.sample = sample.reshape(sample.shape[0], -1)  # All samples
        self.num_sample = sample.shape[0]  # Total number of samples
        self.label = label  # All labels
        self.label_list = np.unique(label)  # List of label
        self.num_label = len(self.label_list)  # Number of labels
        self.sorted_eigenvectors = None
        self.eigenvector_subset = None
        self.save_fig = True

    def reduce_train_dim(self, n_component):

        # Total covariance
        scatter_t = np.cov(self.sample.T) / self.num_sample

        # Within-class scatter
        scatter_w = 0
        for c in range(self.num_label):
            # Find all the corresponding samples for each label
            class_c_idxs = np.flatnonzero(self.label == self.label_list[c])
            scatter_w += np.cov(self.sample[class_c_idxs].T) * len(class_c_idxs)
        scatter_w = scatter_w / self.num_sample

        # Between-class scatter
        scatter_b = scatter_t - scatter_w

        # Get eigenvalues and eigenvectors
        eigen_values, eigen_vectors = np.linalg.eigh(np.dot(np.linalg.inv(scatter_w), scatter_b))

        # Normalize eigenvectors
        # eigen_vectors = eigen_vectors / np.linalg.norm(eigen_vectors, axis=0)

        # Choose the first few components
        sorted_index = np.argsort(eigen_values)[::-1]  # Sort the eigenvalues
        sorted_eigenvectors = eigen_vectors[:, sorted_index]
        self.sorted_eigenvectors = sorted_eigenvectors
        eigenvector_subset = sorted_eigenvectors[:, 0: n_component]
        self.eigenvector_subset = eigenvector_subset

        train_reduced = np.dot(eigenvector_subset.transpose(), self.sample.transpose()).transpose()

        return train_reduced

    def reduce_test_dim(self, test_sample):

        test_sample = test_sample.reshape(-1, 32 * 32)
        test_reduced = np.dot(self.eigenvector_subset.transpose(), test_sample.transpose()).transpose()

        return test_reduced

    def plot_distribution(self, sample_reduced):

        color_map = plt.get_cmap('tab20c')
        colors = [color_map(idx) for idx in np.linspace(0, 1, self.num_label - 1)]
        colors.append('red')
        if sample_reduced.shape[-1] == 2:
            plt.figure(figsize=(8, 6))
            for idx, c in enumerate(self.label_list):
                if c == 'selfie':
                    marker = '*'
                else:
                    marker = 'o'
                class_c_idxs = np.flatnonzero(self.label == c)
                plt.scatter(sample_reduced[class_c_idxs, 0], sample_reduced[class_c_idxs, 1],
                            label=c, c=colors[idx], marker=marker)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('Two dimensions')
            plt.legend(loc=1, fontsize=9, bbox_to_anchor=(1.2, 1))

        elif sample_reduced.shape[-1] == 3:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(projection='3d')
            for idx, c in enumerate(self.label_list):
                if c == 'selfie':
                    marker = '*'
                else:
                    marker = 'o'
                class_c_idxs = np.flatnonzero(self.label == c)
                ax.scatter(sample_reduced[class_c_idxs, 0], sample_reduced[class_c_idxs, 1],
                            sample_reduced[class_c_idxs, 2], label=c, c=colors[idx], marker=marker)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.title.set_text('Three dimensions')

        if self.save_fig:
            plt.tight_layout()
            plt.savefig('../result/LDA_{}-dimensions.png'.format(sample_reduced.shape[-1]),
                        dpi=200, bbox_inches='tight')
