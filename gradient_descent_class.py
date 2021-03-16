import numpy as np


class GradientDescent:
    def __init__(self,
                 data_params=(4, 0.4, 1000),
                 step=0.1,
                 loss='hinge'):
        self.mu, self.d, self.n = data_params
        self.data = self.generate_data()
        self.x_train, self.x_test, self.y_train, self.y_test = self.test_train_split()
        self.theta_init = np.random.uniform(0.0, 1.0, size=self.d)
        self.tuning_param = self.set_loss_function(loss)

    def generate_data(self):
        mu, d, n = self.mu, self.d, self.n
        mean_a = [mu] + [0] * (d - 1)  # mean vector
        mean_b = [-mu] + [0] * (d - 1)
        cov = np.identity(d)  # identity covariance
        x_vals_a = np.random.multivariate_normal(mean_a, cov, int(n / 2))
        x_vals_b = np.random.multivariate_normal(mean_b, cov, int(n / 2))
        y_vals = np.random.choice([-1, 1], size=(n, 1))
        x_vals = np.concatenate((x_vals_a, x_vals_b), axis=0)
        return np.concatenate((x_vals, y_vals), axis=1)

    def test_train_split(self, split=0.7):
        data = self.data
        n = self.n
        x = np.asarray([a[:-1] for a in data])
        x_train, x_test = x[:int(n * split)], x[int(n * int(1 - split)):]

        y = np.asarray([a[-1] for a in data])
        y_train, y_test = y[:int(n * split)], y[int(n * int(1 - split)):]
        return x_train, x_test, y_train, y_test

    def set_loss_function(self, loss_name):
        loss_functions = {'hinge': lambda x: max(0, 1 + x),
                          'exp': lambda x: np.exp(x),
                          'logistic': lambda x: np.log2(1 + np.exp(x))}
        return loss_functions[loss_name]


if __name__ == '__main__':
    data_dims = (5, 3, 200)  # dimensions of the data distribution
    step_size = 0.1
    gd = GradientDescent(data_dims, step_size, 'hinge')
