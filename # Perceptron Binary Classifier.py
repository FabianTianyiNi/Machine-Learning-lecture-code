# Perceptron Binary Classifier
# Make a prediction with the weights and bias

# This is training data
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]

weights = [-0.1, 0.20653640140000007, -0.23418117710000003]


# for row in dataset:
# 	prediction = predict(row, weights)
# 	print 'Expected = %d, predicted = %d' %(row[-1], prediction)

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += row[i] * weights[i + 1]
    return 1.0 if activation >= 0.0 else 0.0


# Estimate the Percepron weights using stochastic graient descent
def train_weights(train_data, l_rate, n_epoch):
    weights = [0.0 for i in xrange(len(train_data[0]))]
    for epoch in xrange(n_epoch):
        sum_of_error = 0
        for row in train_data:
            predictions = predict(row, dataset)
            sum_of_error += abs(predictions - row[-1])
            weights[0] = weights[0] + l_rate * sum_of_error
            for i in xrange(len(row) - 1):
                weights[i + 1] = weights[i + 1] + l_rate * sum_of_error
    print epoch, sum_of_error
    print weights


l_rate = 0.1
n_epoch = 5
train_weights(dataset, 0.1, 5)
