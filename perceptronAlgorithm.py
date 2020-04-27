
# Making a prediction with weight calculation
def prediction(row, weights, bias):
    """
    activation = sum(weight + training data) + bias

    1.0 => activation >= 0.0
    0.0 => activation < 0.0
    """

    sum = bias

    for i in range(len(row) - 1):
        #print(i, row[i], activation, weights[i + 1])
        sum += weights[i + 1] * row[i]
    print(sum)
    return 1.0 if sum >= 0.0 else 0.0


# test predictions
dataset = [[2.7810836, 2.550537003, 0]#,
          # [1.465489372, 2.362125076, 0],
          # [3.396561688, 4.400293529, 0],
           #[1.38807019, 1.850220317, 0],
           #[3.06407232, 3.005305973, 0],
           #[7.627531214, 2.759262235, 1],
           #[5.332441248, 2.088626775, 1],
           #[6.922596716, 1.77106367, 1],
           #[8.675418651, -0.242068655, 1],
           #[7.673756466, 3.508563011, 1]
           ]
weights = [-0.1, 0.20653640140000007, -0.23418117710000003]
bias = -0.1

for row in dataset:
    predicted = prediction(row, weights, bias)
    #print("Expected=%d, Predicted=%d" % (row[-1], predicted))