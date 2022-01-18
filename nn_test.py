import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dense import Dense
from activations import Tanh, Sigmoid, ReLu, lReLu
from losses import mse, mse_prime
from network import train, predict
from nn_test_data import simple_data_sets,\
     harder_data_sets,\
     challenging_data_sets,\
     manual_weight_data_sets,\
     all_data_sets

"""
1++
0-+
 01
"""
# or_data = ((0,0,0),
#            (0,1,1),
#            (1,0,1),
#            (1,1,1),
#            (0.25,0,0),
#            (0,0.25,0))
# X = np.reshape(np.asarray(or_data)[:, :-1], (6, 2, 1))
# Y = np.reshape(np.asarray(or_data)[:, -1:], (6, 1, 1))

# X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
# Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# test_data_sets = harder_data_sets
# test_data_sets = challenging_data_sets
# test_data_sets = manual_weight_data_sets
test_data_sets = all_data_sets
np.random.seed(0)
total_correct = 0
total_tests = 0
for data_sets in test_data_sets:
    correct = 0
    print("Processing", data_sets[0], "...")
    train_data = data_sets[1]
    test_data = data_sets[2]
    length = len(train_data)
    X = np.reshape(np.asarray(train_data)[:, :-1], (length, 2, 1))
    Y = np.reshape(np.asarray(train_data)[:, -1:], (length, 1, 1))

    network = [
        Dense(2, 3),
        Sigmoid(),
        # Tanh(),
        # lReLu(),
        Dense(3, 3),
        # Sigmoid(),
        lReLu(),
        Dense(3, 1),
        Sigmoid()
        # Tanh()
        # ReLu()
    ]

    # train
    train(network, mse, mse_prime, X, Y, epochs=10000, learning_rate=0.1)

    # print(predict(network, np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))))
    length = len(test_data)
    X = np.reshape(np.asarray(test_data)[:, :-1], (length, 2, 1))
    Y = np.reshape(np.asarray(test_data)[:, -1:], (length, 1, 1))
    for x, y in zip(X, Y):
        result = predict(network, x)[0][0]
        rounded_result = round(result)
        if rounded_result == y:
            correct += 1
            print("test(%s) returned: %s => %s [%s]" %(str(x),
                                                        str(result),
                                                        rounded_result,
                                                        "correct"))
        else:
            print("test(%s) returned: %s => %s [%s]" %(str(x),
                                                        str(result),
                                                        rounded_result,
                                                        "wrong"))
    total_correct += correct
    total_tests += len(test_data)                                                   
    print("Accuracy: %f" %(correct/len(test_data)))

print("Final Accuracy of all test sets: %f" %(total_correct/total_tests))

# decision boundary plot
# points = []
# for x in np.linspace(0, 1, 20):
#     for y in np.linspace(0, 1, 20):
#         z = predict(network, [[x], [y]])
#         points.append([x, y, z[0,0]])

# points = np.array(points)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
# plt.show()
