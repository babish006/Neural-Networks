import pandas as pd


# Estimating the values for the weight
def weight_training(trainSet, learningRate, epoch):
    """
    Learning rate: Weight correction rate during updation
    Epochs: Number of execution of weight updation

    w1 = w1 + Learning rate * (expected - predicted) * X1
    bias or w0 = bias + Learning rate * (expected - predicted)

    :return:
    """
    weight = [0.0 for i in range(len(trainSet[0]))]

    for x in range(epoch):
        total_error = 0.0
        for row in trainSet:
            # Predicting the value with existing variables
            predicted_value = prediction(row, weight)

            # Calculating the correctness of prediction
            error = row[-1] - predicted_value
            total_error += error**2

            # Calculating the value for bias
            weight[0] = weight[0] + learningRate * error
            #print(weight[0])
            
            #print(x, row, 'Expected:', row[-1], 'Predicted:', predicted_value, 'Error:', error, 'Total Error:', total_error)

            # Weight for each input attribute
            for i in range(len(row)-1):
                weight[i + 1] = weight[i + 1] + learningRate * error * row[i]

                #print(i, weight[i + 1], learningRate , error , row[i], end='')
                #print('')
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learningRate, total_error))
    return weight

# Making a prediction with weight calculation
def prediction(row, weight):
    """
        activation = sum(weight + training data instances) + bias
    when n=2
        activation = (X1 * w1) + (X2 * w2) + bias

    Weight values received through function
    weight[0] = bias
    weight[1] = w1
    weight[n] = wn

    :return:
    1.0 => activation >= 0.0
    0.0 => activation < 0.0
    """

    sum = weight[0]

    for i in range(len(row) - 1):
        #print(row[i])
        sum += weight[i+1] * row[i]

    return 1.0 if sum >= 0.0 else 0.0


# Test data

# Capturing filename from user
filename = input("Please enter the filename of training dataset: ")
epoch = int(input("EPOCH Value: "))

# Reading train dataset from file using pandas
trainSet = pd.read_csv(filename, delimiter=" ")
trainSet = trainSet.values.tolist()

#print(trainSet)

# Weight calculating variables
learningRate = 0.1


# Deriving the weights from the training dataset
weight = weight_training(trainSet, learningRate, epoch)

print("")

# Predicting the class values using the weights
# received from the training dataset
count = 0

for row in trainSet:
    predicted_value = prediction(row, weight)
    if(row[-1] == predicted_value):
        count += 1
    print("Expected=%d, Predicted=%d" % (row[-1], predicted_value))
print("")
print("Accuracy Rate:",count/len(trainSet)*100)

