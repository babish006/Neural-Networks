import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier

trainFile = input("Please enter the filename of training dataset: ")
testFile = input("Please enter the filename of test dataset: ")

trainSet = pd.read_csv(trainFile, delimiter=" ")

X_train = trainSet.drop('Class',axis=1)
y_train = trainSet['Class']

testSet = pd.read_csv(testFile, delimiter=" ")

X_test = testSet.drop('Class',axis=1)
y_test = testSet['Class']

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Nearest neighbour classifier
knn = KNeighborsClassifier(n_neighbors=19)

print(knn)

knn.fit(X_train,y_train)

predictions = knn.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

print("KNN Test Mean Squared Error =", mean_squared_error(y_test, predictions))

print("KNN Train Accuracy =", knn.score(X_train,y_train))
print("KNN Test Accuracy =",knn.score(X_test,y_test))


