from random import randint
from math import isnan

raw_data0 = [
    [1, 1],
    [2, 2],
    [3, 3]
]

raw_data1 = [
    [00, 00],
    [.22, .22],
    [.24, .58],
    [.33, .20],
    [.37, .55],
    [.44, .39],
    [.44, .54],
    [.57, .53],
    [.93, 1.00],
    [1.00, .61]
]

raw_data2 = [
    [00, 00, 00],
    [.22, .22, .23],
    [.24, .58, .51],
    [.33, .20, .30],
    [.37, .55, .61],
    [.44, .39, 1.00],
    [.44, .54, 1.00],
    [.57, .53, .7],
    [.93, 1.00, 1.1],
    [1.00, .61, .9]
]


def check_data(hypothesis_params, data_features):
    for i in range(len(data_features)):
        return False if len(hypothesis_params) != len(data_features[i]) else True

def hypothesis(theta, x):
    sum_hypothesis = 0
    # Hypothesis = (θ1 * x1) + (θ2 * x2) + (θ3 * x3) + ...
    for i in range(len(theta)):
        sum_hypothesis +=  theta[i] * x[i]
    return sum_hypothesis

def gradient_descent(theta, training_data):
    if not check_data(theta, training_data):
        return "ERROR: Hypothesis parameters != Training data features"

    rate = .05

    # Insert 1 to x(0) so that way:
    # a + b(x[1]) = a(x[0]) + b(x[1]) AND
    # (hθ(x[i]) - y[i]) = (hθ(x[i]) - y[i]) * x[0]
    for i in range(len(training_data)):
        training_data[i].insert(0, 1)

    # Begin gradient descent
    while True:
        derivatives = [[_ for _ in range(len(training_data))] for _ in range(len(theta))]
        new_theta = [_ for _ in range(len(theta))]
        for j in range(len(training_data[0])-1):
            for i in range(len(training_data)):
                h = hypothesis(theta, training_data[i])
                xi = training_data[i][j]
                yi = training_data[i][-1]

                derivatives[j][i] = (h - yi) * xi

            new_theta[j] = round((theta[j] - (rate * sum(derivatives[j], 0))), 5)

        if new_theta == theta:
            return new_theta
        elif isnan(new_theta[0]):
            return "ERROR: Rate too large"
        else:
            theta = new_theta

def test():
    # Hyp length must equal the length
    # of each training set of the data
    a = randint(-100, 100)
    b = randint(-100, 100)
    c = randint(-100, 100)
    hyp = [
        a,
        b,
        c
    ]
    print(gradient_descent(hyp, raw_data2))

test()