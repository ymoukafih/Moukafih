import numpy as np
from sklearn import datasets
from numpy.linalg import inv, linalg


class LDA:

    def LDA_Calculation(self, features, labels, n_components):
        mean_features = np.mean(features, axis=0)
        S_W = np.zeros([features.shape[1], features.shape[1]])
        S_B = np.zeros([features.shape[1], features.shape[1]])
        for label in np.unique(labels):
            class_features = features[label == labels]
            mean_class = np.mean(class_features, axis=0)
            average_w = class_features - mean_class
            S_W += np.matmul(average_w.T, average_w)
            average_B = mean_class - mean_features
            average_B = average_B.reshape(-1, 1)
            S_B += class_features.shape[0]*average_B.dot(average_B.T)

        S_W_inv = inv(S_W)
        S_W_inv_S_B = S_W_inv.dot(S_B)
        eigValues, eigVectors = linalg.eig(S_W_inv_S_B)
        index = np.argsort(eigValues)[::-1]
        eigVectors = eigVectors.T[index]
        return np.matmul(features, eigVectors[:n_components].T)



data = datasets.load_iris()
X = data.data
y = data.target

lda = LDA()
projected = lda.LDA_Calculation(X, y, 2)

print("representations on the new space:", projected[:10])
