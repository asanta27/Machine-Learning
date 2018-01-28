from random import randint

training_data0 = [
    (1, 1),
    (2, 2),
    (3, 3)
]

training_data1 = [
    (0.00, 0.00),
    (0.22, 0.22),
    (0.24, 0.58),
    (0.33, 0.20),
    (0.37, 0.55),
    (0.44, 0.39),
    (0.44, 0.54),
    (0.57, 0.53),
    (0.93, 1.00),
    (1.00, 0.61)
]


def gradient_descent(a, b, raw_data):
    counter = 0
    while True:
        rate = .01
        da = 0
        db = 0
        for coordinates in raw_data:
            xi = coordinates[0]
            yi = coordinates[1]
            # Hypothesis
            ho = a + b * xi

            # Partial derivative of a(J)
            da += -(yi - ho)
            # Partial derivative of b(J)
            db += -(yi - ho) * xi

        new_a = a - rate * da
        new_b = b - rate * db

        # Repeat until convergence
        if new_a == a and new_b == b:
            new_a = round(new_a, 2)
            new_b = round(new_b, 2)
            new_a = 0.0 if new_a == -0 else new_a
            new_b = 0.0 if new_b == -0 else new_b
            return new_a, new_b
        else:
            counter +=1
            a = new_a
            b = new_b


def run():
    # Hypothesis Generator
    ho_a = randint(-100, 100)
    ho_b = randint(-100, 100)
    a = str(gradient_descent(ho_a, ho_b, training_data1)[0])
    b = str(gradient_descent(ho_a, ho_b, training_data1)[1])
    string = "Line of best fit: y = " + a + " + " + b + "x"
    print(string)


run()
