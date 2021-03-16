import numpy as np


class GradientDescent:
    def __init__(self, data_params=(4, 0.4, 1000)):
        self.d = data_params[0]  # dimensions
        self.mu = data_params[1]
        self.n = data_params[2]
        self.data = self.generate_data()
        self.x_train, self.x_test, self.y_train, self.y_test = self.test_train_split()

    def generate_data(self):
        mu = self.mu
        d = self.d
        n = self.n
        mean_a = [mu] + [0] * (d - 1)  # mean vector
        mean_b = [-mu] + [0] * (d - 1)
        cov = np.identity(d)  # identity covariance
        x_vals_a = np.random.multivariate_normal(mean_a, cov, int(n / 2))
        x_vals_b = np.random.multivariate_normal(mean_b, cov, int(n / 2))
        y_vals = np.random.choice([-1, 1], size=(n, 1))
        x_vals = np.concatenate((x_vals_a, x_vals_b), axis=0)
        return np.concatenate((x_vals, y_vals), axis=1)

    def test_train_split(self):
        data = self.data
        x = np.asarray([a[:-1] for a in data])
        x_train, x_test = x[:int(self.n * 0.7)], x[int(self.n * 0.3):]

        y = np.asarray([a[-1] for a in data])
        y_train, y_test = y[:int(self.n * 0.7)], y[int(self.n * 0.3):]
        return x_train, x_test, y_train, y_test




if __name__ == '__main__':
    data_dims = (5, 3, 200)  # dimensions of the data distrubution
    gd = GradientDescent(data_dims)
