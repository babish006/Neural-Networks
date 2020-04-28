import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix



trainSet = pd.read_csv('wine_training', delimiter=" ")

X_train = trainSet.drop('Class',axis=1)
y_train = trainSet['Class']

testSet = pd.read_csv('wine_test', delimiter=" ")

X_test = testSet.drop('Class',axis=1)
y_test = testSet['Class']

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=1000)

mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))





