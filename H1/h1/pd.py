import pandas
import numpy

from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 

numberOfSelectedFeatures = [5,10,40,100,250,700]
data = loadarff('pd_speech.arff.txt')
dataFrame = pandas.DataFrame(data[0])

# Features of the input data
X = numpy.array(dataFrame.iloc[:, :-1])
# Target vector of the input data
y = numpy.array(dataFrame.iloc[:, -1])

# Each b'0' value in the target is converted to 0 and each b'1' value is
# converted to 1, so that the target y can be used in the mutual_info_classif function
y = numpy.where(y == y[0], 1, 0)

# Mutual information between each of the features and the target
discriminativePower = mutual_info_classif(X, y, random_state=1)

# Vector with the index of each variable by decreasing discriminative power order
idx = (-discriminativePower).argsort()[:752]

testingAccuracy = []
trainingAccuracy = []

for index in numberOfSelectedFeatures:
    selectedX = X[:, idx[:index]]
    X_train, X_test, y_train, y_test = train_test_split(selectedX, y, test_size=0.3, stratify=y, random_state=1)
    tree = DecisionTreeClassifier(max_depth=None, random_state=1)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    testingAccuracy.append(accuracy_score(y_test, y_pred))
    y_pred = tree.predict(X_train)
    trainingAccuracy.append(accuracy_score(y_train, y_pred))

plt.plot(numberOfSelectedFeatures, testingAccuracy) 
plt.xlabel('x - number of selected features') 
plt.ylabel('y - testing accuracy') 
plt.show() 

plt.plot(numberOfSelectedFeatures, trainingAccuracy) 
plt.xlabel('x - number of selected features') 
plt.ylabel('y - training accuracy') 
plt.show()
