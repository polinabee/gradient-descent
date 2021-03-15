'''
Implementation of gradient descent
Source: https://medium.com/@DataStevenson/pokemon-stats-and-gradient-descent-for-multiple-variables-c9c077bbf9bd
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def hypothesis(x, theta):
    return np.dot(
        np.transpose(theta),
        x
    )


def RMSE(y, y_hat):
    return np.sqrt(sum((y - y_hat) ** 2) / len(y))


def gradientDescent(x, y, x_test, y_test, theta, alpha):
    RMSEs = []
    m = len(x)
    rmse = 900

    while rmse > 100:
        for j in range(len(theta)):
            gradient = 0
            for i in range(m):
                gradient += (hypothesis(x[i], theta) - y[i]) * x[i][j]
            gradient *= 1 / m
            theta[j] = theta[j] - (alpha * gradient)
        y_pred = [hypothesis(x, theta) for x in x_test]
        rmse = RMSE(y_pred, y_test)
        RMSEs.append(rmse)
    return theta, RMSEs


def generateZValues(x, theta):
    z_values = []
    for i in range(len(x)):
        z_values.append(hypothesis(x[i], theta))
    return np.asarray(z_values)


if __name__ == '__main__':
    df = pd.read_csv('pokemon_alopez247.csv')

    total = np.asarray(df['Total'])
    special_attack = np.asarray(df['Sp_Atk'])
    catch_rate = np.asarray(df['Catch_Rate'])

    x = np.asarray([[tot, spec_atk] for tot, spec_atk in zip(total, special_attack)])
    x_train, x_test = x[:int(len(x) * 0.7)], x[int(len(x) * 0.3):]

    y = np.asarray([rate for rate in catch_rate])  # Gets our output
    y_train, y_train = y[:int(len(y) * 0.7)], y[int( len(y) * 0.3):]  # Splits our outputs in half between training and testing
    theta_init = np.random.uniform(0.0, 1.0, size=2)

    alpha = 0.00001
    n = 350
    thetas, rmses = gradientDescent(x_train,
                                y_train,
                                x_test,
                                y_train,
                                theta_init,
                                alpha)

    plt.plot(rmses)
    plt.title(f'RMSE convergence over {len(rmses)} iterations')
    plt.show()
