import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import keras
import tensorflow
from keras import Sequential
from keras.layers import Dense

# Create a model without initializing the weight


def create_model_null():
    model = Sequential()
    model.add(Dense(68, input_dim=34, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Create a model with assigned weight


def create_model(w1, b1, w2, b2):
    L = [w1, b1, w2, b2]
    np.asarray(L)
    model = Sequential()
    model.set_weights(L)
    model.add(Dense(68, input_dim=34, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Create a random weight set for the Neural Network


def create_initial_random():
    weight1 = np.random.rand(34, 68)
    bias1 = np.random.rand(1, 68)
    weight2 = np.random.rand(68, 1)
    bias2 = np.random.rand(1, 1)
    return weight1, bias1, weight2, bias2

# Create a random velocity for the PSO algorithm


def create_initial_velocity():
    velocity_w1 = np.random.rand(34, 68)
    velocity_b1 = np.random.rand(1, 68)
    velocity_w2 = np.random.rand(68, 1)
    velocity_b2 = np.random.rand(1, 1)
    return velocity_w1, velocity_b1, velocity_w2, velocity_b2

# Self-created Sigmoid function


def sigmoid(inpt):
    return 1.0 / (1.0 + np.exp(-1 * inpt))

# Self-defined Fitness function, return the accuracy of the NN model


def fitness_function(X, target, w1, b1, w2, b2):
    m = create_model(w1, b1, w2, b2)
    m.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=['accuracy'])
    accr = m.test_on_batch(X, target, sample_weight=None,
                           reset_metrics=True)[1]
    return accr


# Main function
if __name__ == "__main__":
    df = pd.read_csv('/Users/juezhang/Desktop/Assignment3/ionosphere.csv')
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    # Paraphase the Char target to Int
    target = []
    for val in Y:
        if val == 'b':
            target.append([0])
        else:
            target.append([1])
    target = np.asarray(target)
    target.transpose()
    weight1 = []
    bias1 = []
    weight2 = []
    bias2 = []
    velocity = []
    local_optimal = []
    local_optimal_score = []
    global_optimal = []
    global_optimal_score = 0
    for count in range(10):
        w1, b1, w2, b2 = create_initial_random()
        weight1.append(w1)
        bias1.append(b1)
        weight2.append(w2)
        bias2.append(b2)
        vw1, vb1, vw2, vb2 = create_initial_velocity()
        temp = [vw1, vb1, vw2, vb2]
        velocity.append(temp)
        temp = [w1, b1, w2, b2]
        local_optimal.append(temp)
        cur_score = fitness_function(X, target, w1, b1, w2, b2)
        local_optimal_score.append(cur_score)
        if cur_score > global_optimal_score:
            global_optimal = temp
            global_optimal_score = cur_score
    history_best = []
    for itr in range(800):
        for i in range(10):
            # np.add  np.subtract *np.array
            # weight1 velocity update
            velocity[i][0] = np.array(velocity[i][0]) + 2 * np.random.normal(0, 1, 1) * np.subtract(
                np.array(local_optimal[i][0]), np.array(weight1[i]))
            velocity[i][0] = velocity[i][0] + 2 * np.random.normal(0, 1, 1) * \
                np.subtract(np.array(global_optimal[0]), np.array(weight1[i]))
            # bias1 velocity update
            velocity[i][1] = np.array(velocity[i][1]) + 2 * np.random.normal(0, 1, 1) * np.subtract(
                np.array(local_optimal[i][1]), np.array(bias1[i]))
            velocity[i][1] = velocity[i][1] + 2 * np.random.normal(0, 1, 1) * \
                np.subtract(np.array(global_optimal[1]), np.array(bias1[i]))
            # weight2 velocity update
            velocity[i][2] = np.array(velocity[i][2]) + 2 * np.random.normal(0, 1, 1) * np.subtract(
                np.array(local_optimal[i][2]), np.array(weight2[i]))
            velocity[i][2] = velocity[i][2] + 2 * np.random.normal(0, 1, 1) * \
                np.subtract(np.array(global_optimal[2]), np.array(weight2[i]))
            # bias2 velocity update
            velocity[i][3] = np.array(velocity[i][3]) + 2 * np.random.normal(0, 1, 1) * np.subtract(
                np.array(local_optimal[i][3]), np.array(bias2[i]))
            velocity[i][3] = velocity[i][3] + 2 * np.random.normal(0, 1, 1) * \
                np.subtract(np.array(global_optimal[3]), np.array(bias2[i]))
            # update location
            weight1[i] += velocity[i][0]
            bias1[i] += velocity[i][1]
            weight2[i] += velocity[i][2]
            bias2[i] += velocity[i][3]
            # Get the fitness score for new location and check for optimal update
            cur_score = fitness_function(
                X, target, weight1[i], bias1[i], weight2[i], bias2[i])
            if local_optimal_score[i] < cur_score:
                local_optimal_score[i] = cur_score
                local_optimal[i] = [weight1[i], bias1[i], weight2[i], bias2[i]]
            if global_optimal_score < cur_score:
                global_optimal_score = cur_score
                global_optimal = [weight1[i], bias1[i], weight2[i], bias2[i]]
        print(global_optimal_score, itr)
        history_best.append(global_optimal_score)
    plt.plot(history_best)
    plt.show()
