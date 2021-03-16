import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_data(a, d, n):
    mean_a = [a] + [0] * (d - 1)  # mean vector
    mean_b = [-a] + [0] * (d - 1)
    cov = np.identity(d)  # identity covariance
    x_vals_a = np.random.multivariate_normal(mean_a, cov, int(n / 2))
    x_vals_b = np.random.multivariate_normal(mean_b, cov, int(n / 2))
    y_vals = np.random.choice([-1, 1], size=(n, 1))
    x_vals = np.concatenate((x_vals_a, x_vals_b), axis=0)
    return np.concatenate((x_vals, y_vals), axis=1)


def dot_product(x, w):
    return w.T @ x


def RMSE(y, y_hat):
    return np.sqrt(sum((y - y_hat) ** 2) / len(y))


def gradientDescent(x, y, x_test, y_test, theta, alpha, phi, rmse_cutoff):
    RMSEs = []
    m = len(x)
    rmse = RMSE([dot_product(x, theta) for x in x_test], y_test)
    iterations = 0

    while rmse > rmse_cutoff and iterations < 3000:
        print(theta)
        iterations += 1
        for j in range(len(theta)):
            gradient = 0
            for i in range(m):
                gradient += phi(dot_product(x[i], theta) - y[i]) * x[i][j]
            gradient *= 1 / m
            theta[j] = theta[j] - (alpha * gradient)
        y_pred = [dot_product(x, theta) for x in x_test]
        rmse = RMSE(y_pred, y_test)
        RMSEs.append(rmse)
    return theta, RMSEs


if __name__ == '__main__':
    d = 4  # dimensions
    a = 0.4  # mean
    n = 1000  # number of points

    data = generate_data(a, d, n)

    x = np.asarray([a[:-1] for a in data])
    x_train, x_test = x[:int(len(x) * 0.7)], x[int(len(x) * 0.3):]

    y = np.asarray([a[-1] for a in data])
    y_train, y_train = y[:int(len(y) * 0.7)], y[int(len(y) * 0.3):]

    theta_init = np.random.uniform(0.0, 1.0, size=d)

    tuning_param = 0.001

    loss_functions = {'hinge': lambda x: max(0, 1 + x),
                      'exp': lambda x: np.exp(x),
                      'logistic': lambda x: np.log2(1 + np.exp(x))}

    thetas, error = gradientDescent(x_train,
                                    y_train,
                                    x_test,
                                    y_train,
                                    theta_init,
                                    tuning_param,
                                    loss_functions['hinge'],
                                    0.7)

    plt.plot(error)
    plt.title(f'RMSE convergence over {len(error)} iterations')
    plt.show()
