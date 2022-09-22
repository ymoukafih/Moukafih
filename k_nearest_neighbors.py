import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def accuracy_score(y_true, y_pred):
    return round(float(sum(y_true == y_pred))/float(len(y_true)) * 100, 2)


class KNN:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def KNN_calculation(self, test, k):
        output = []
        for idx in range(test.shape[0]):
            values = np.array([np.sum((test[idx] - self.features[i])**2) for i in range(self.features.shape[0])])
            output.append(np.bincount(self.labels[np.argsort(values)[:k]]).argmax())
        return np.array(output)


data = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=0)

knn = KNN(X_train, y_train)
for k in range(1, 10):
    print(f"The accuracy score for {k} = {accuracy_score(knn.KNN_calculation(X_test, k), y_test)}")