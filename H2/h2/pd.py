import scipy
import pandas
import numpy

from scipy.io.arff import loadarff
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt


data = loadarff('pd_speech.arff.txt')
dataFrame = pandas.DataFrame(data[0])
# Features of the input data
X = numpy.array(dataFrame.iloc[:, :-1])
# Target vector of the input data
y = numpy.array(dataFrame.iloc[:, -1])
# Each b'0' value in the target is converted to 0 and each b'1' value is converted to 1
y = numpy.where(y == y[0], 1, 0)
folds = StratifiedKFold(n_splits=10, shuffle = True, random_state=0)


# kNN
neigh = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski')
cumulativeTestingConfusionMatrix = numpy.array([[0, 0], [0, 0]])
t_statistic = 0
# Generates indices so data is split into training and test set and iterates the 10 splits
for train_index, test_index in folds.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    numpy.add(cumulativeTestingConfusionMatrix, cm, cumulativeTestingConfusionMatrix)
    t_statistic += ttest_rel(y_pred, y_test, axis=0)[1] 
t_statistic /= 10
accuracy = (cumulativeTestingConfusionMatrix[0][0] + cumulativeTestingConfusionMatrix[1][1]) / numpy.sum(cumulativeTestingConfusionMatrix)
print("kNN p-value: ", t_statistic)
print("kNN Confusion Matrix Accuracy: ", accuracy)
disp = ConfusionMatrixDisplay(confusion_matrix = cumulativeTestingConfusionMatrix)
disp = disp.plot()
plt.title('Confusion matrix for kNN')
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.show()


# Gaussian Naive Bayes
gnb = GaussianNB()
cumulativeTestingConfusionMatrix = numpy.array([[0, 0], [0, 0]])
t_statistic = 0
for train_index, test_index in folds.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    numpy.add(cumulativeTestingConfusionMatrix, cm, cumulativeTestingConfusionMatrix)
    t_statistic += ttest_rel(y_pred, y_test, axis=0)[1]
t_statistic /= 10
accuracy = (cumulativeTestingConfusionMatrix[0][0] + cumulativeTestingConfusionMatrix[1][1]) / numpy.sum(cumulativeTestingConfusionMatrix)
print("Gaussian Naive Bayes p-value: ", t_statistic)
print("Gaussian Naive Bayes Confusion Matrix Accuracy: ", accuracy)
disp = ConfusionMatrixDisplay(confusion_matrix = cumulativeTestingConfusionMatrix)
disp = disp.plot()
plt.title('Confusion matrix for Gaussian Naive Bayes')
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.show()
