from bin.NeuralNet import *
from bin.functions import loss, derivative
import matplotlib.pyplot as plt


class BackProp(NeuralNet):
    def __init__(self, config: tuple = None, file_path: str = None):
        super().__init__(config, file_path)
        self.learn_rate = 0.1
        self.epochs = 100

    def learning(self, x, plot: bool = True):
        plot_err = list()
        for epoch in range(self.epochs):
            err = list()
            for batch in x:
                z = self.inference(batch[0].copy())
                err.append(loss(batch[1].copy(), z))
                dt = derivative(loss, "y", z, x=batch[1].copy())
                dh = Vector()
                for i in range(len(self._config) - 2, -1, -1):
                    if i != len(self._config) - 2:
                        dt = dh * self._t[i].in_func(derivative, "val", var="x", function=self.activation_func)
                    self._b[i] -= self.learn_rate * dt
                    dw = self._h[i].copy().transpose() @ dt
                    self._w[i] -= self.learn_rate * dw
                    if i != 0:
                        dh = Vector().from_matrix(dt @ self._w[i].copy().transpose())
            mean = sum(err) / len(err)
            plot_err.append(mean)
            if (epoch + 1) % self.print_every == 0:
                print(f"Epoch {epoch + 1} loss: {mean}")
            self.save()
        if plot:
            fig, ax = plt.subplots()
            ax.plot(plot_err)
            plt.show()


class ResilientProp(NeuralNet):
    def __init__(self, config: tuple = None, file_path: str = None, delta_start=0.1, gamma_start=0.1):
        super().__init__(config, file_path)
        self.epochs = 100
        self.correction_plus = 1.2
        self.correction_minus = 0.5
        self.correction_max = 50
        self.correction_min = 1e-6
        self._delta = list()
        self._gamma = list()
        for i in range(1, len(self._config)):
            self._delta.append(Matrix(self._config[i - 1], self._config[i], delta_start))
            self._gamma.append(Vector(self._config[i], gamma_start))

    def learning(self, x, plot: bool = True):
        plot_err = list()
        if not 0 < self.correction_minus < 1 < self.correction_plus:
            raise ValueError
        for epoch in range(self.epochs):
            err = list()
            dws = [list(), list()]
            dts = [list(), list()]
            ind = 0
            for batch in x:
                z = self.inference(batch[0])
                err.append(loss(batch[1], z))
                dt = derivative(loss, "y", z, x=batch[1])
                dh = Vector()
                count = 0
                for i in range(len(self._config) - 2, -1, -1):
                    if i != len(self._config) - 2:
                        dt = dh * self._t[i].in_func(derivative, "val", var="x", function=self.activation_func)
                    dt_size = len(dt)
                    dts[ind % 2].append(dt)
                    if ind != 0:
                        for j in range(dt_size):
                            if dts[ind % 2][count][j] * dts[(ind + 1) % 2][count][j] > 0:
                                self._gamma[i][j] *= self.correction_plus
                            else:
                                self._gamma[i][j] *= self.correction_minus
                    for j in range(dt_size):
                        db = dt[j]
                        if db > 0:
                            self._b[i][j] -= self._gamma[i][j]
                        elif db < 0:
                            self._b[i][j] += self._gamma[i][j]
                    dw = self._h[i + 1].copy().transpose() @ dt
                    dws[ind % 2].append(dw)
                    n, m = dw.get_size()
                    if ind != 0:
                        for j in range(n):
                            for k in range(m):
                                if dws[ind % 2][count][j][k] * dws[(ind + 1) % 2][count][j][k] > 0:
                                    self._delta[i][j][k] *= self.correction_plus
                                else:
                                    self._delta[i][j][k] *= self.correction_minus
                    for j in range(n):
                        line = dw[j]
                        for k in range(m):
                            d_elem = line[k]
                            if d_elem > 0:
                                self._w[i][j][k] -= self._delta[i][j][k]
                            elif d_elem < 0:
                                self._w[i][j][k] += self._delta[i][j][k]
                    if i != 0:
                        dh = dt @ self._w[i].copy().transpose()
                    count += 1
                ind += 1
            mean = sum(err) / len(err)
            plot_err.append(mean)
            if (epoch + 1) % self.print_every == 0:
                print(f"Epoch {epoch + 1} loss: {mean}")
            self.save()
        if plot:
            fig, ax = plt.subplots()
            ax.plot(plot_err)
            plt.show()
