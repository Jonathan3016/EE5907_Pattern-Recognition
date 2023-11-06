import numpy as np
import matplotlib.pyplot as plt

class PCA:

    def __init__(self, X):
        self.X = X.reshape(X.shape[0], -1)
        self.save_fig = True  # Whether the picture is saved or not
        self.X_mean = None  # use for data reconstruction
        self.sorted_eigenvectors = None  # use for plot pc (eigenfaces)
        self.eigenvector = None
        self.threshold = 0.95  # Use for handling of unspecified dimensions
        self.x_recon = None  # use for data reconstruction

    def reduce_train_dim(self, n_component=None):

        # Get the centered data matrix
        X_mean = self.X - np.mean(self.X, axis=0)
        self.X_mean = X_mean
        # Calculate the Covariance Matrix
        cov_mat = np.cov(X_mean, rowvar=False)
        # Compute the eigenvectors
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
        sorted_index = np.argsort(eigen_values)[::-1]  # Sort the eigenvalues
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]  # Sort the eigenvectors base on the eigenvalues
        self.sorted_eigenvectors = sorted_eigenvectors

        # Handling of unspecified dimensions
        if n_component == None:
            for n_component in range(len(sorted_eigenvalue)):
                if sum(sorted_eigenvalue[:n_component]) / sum(sorted_eigenvalue) > self.threshold:
                    break
            print('Choose p={} based on 0.95 of variation to retain.'.format(n_component))

        # Choose the first few components
        eigenvector_subset = sorted_eigenvectors[:, 0: n_component]  # Shape = [1024, n_component]
        self.eigenvector = eigenvector_subset
        # Transform the data
        X_reduced = np.dot(eigenvector_subset.transpose(), X_mean.transpose()).transpose()

        return X_reduced

    def reduce_test_dim(self, test_x):

        test_x = test_x.reshape(-1, 32 * 32)
        X_meaned = test_x - np.mean(test_x, axis=0)
        X_reduced = np.dot(self.eigenvector.transpose(), X_meaned.transpose()).transpose()

        return X_reduced

    def plot_distribution(self, X_reduced):

        figsize = (6, 6)
        if X_reduced.shape[-1] == 2:
            plt.figure(figsize=figsize)
            plt.scatter(X_reduced[:-7, 0], X_reduced[:-7, 1], label='CMU PIE')
            plt.scatter(X_reduced[-7:, 0], X_reduced[-7:, 1], label='SELFIES')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('Two dimensions')
        elif X_reduced.shape[-1] == 3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(projection='3d')
            ax.scatter(X_reduced[:-7, 0], X_reduced[:-7, 1], X_reduced[:-7, 2], label='CMU PIE')
            ax.scatter(X_reduced[-7:, 0], X_reduced[-7:, 1], X_reduced[-7:, 2], label='SELFIES')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.title.set_text('Three dimensions')

        plt.legend()
        if self.save_fig:
            plt.tight_layout()
            plt.savefig('../result/PCA_{}-dimensions.png'.format(X_reduced.shape[-1]), dpi=200, bbox_inches='tight')

    def plot_eigenfaces(self, num_samples=3):

        fig = plt.figure(figsize=(9, 3), dpi=200)
        for i in range(num_samples):
            ax = fig.add_subplot(1, 3, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(self.sorted_eigenvectors[:, i].reshape(32, 32), cmap='gray')
        plt.subplots_adjust(wspace=0, hspace=0)
        if self.save_fig:
            plt.tight_layout()
            plt.savefig('../result/PCA_eigenfaces.png')