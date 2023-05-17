from bin.NeuralNet import *
import matplotlib.pyplot as plt


def loss(x, y):
    return np.sum((x - y) ** 2) / len(x)


class BackProp(NeuralNet):
    def __init__(self, config: tuple = None, file_path: str = None):
        super().__init__(config, file_path)
        self.learn_rate = 0.01
        self.epochs = 1000

    def derivative(self, function, val):
        return (function(val + self.step) - function(val - self.step)) / (2 * self.step)

    def learning(self, x, plot: bool = True):
        plot_err = list()
        for epoch in range(self.epochs):
            err = list()
            for batch in x:
                z = self.inference(batch[0])
                err.append(loss(batch[1], z))
                dh = (loss(batch[1], z + self.step) - loss(batch[1], z - self.step)) / (2 * self.step)
                for i in range(len(self.config) - 2, -1, -1):
                    dt = dh * self.derivative(activation, self.t[i])
                    self.b[i] -= self.learn_rate * dt
                    dw = self.h[i + 1].transpose() @ dt
                    self.w[i] -= self.learn_rate * dw
                    dh = dt @ self.w[i].transpose()
            mean = np.mean(err)
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
        self.epochs = 500
        self.correction_plus = 1.2
        self.correction_minus = 0.5
        self.correction_max = 50
        self.correction_min = 1e-6
        self.delta = list()
        self.gamma = list()
        for i in range(1, len(self.config)):
            self.delta.append(np.full((self.config[i - 1], self.config[i]), delta_start))
            self.gamma.append(np.full(self.config[i], gamma_start))

    def derivative(self, function, val):
        return (function(val + self.step) - function(val - self.step)) / (2 * self.step)

    def learning(self, x, plot: bool = True):
        plot_err = list()
        if not 0 < self.correction_minus < 1 < self.correction_plus:
            raise
        for epoch in range(self.epochs):
            err = list()
            dws = [None, None]
            dts = [None, None]
            ind = 0
            for batch in x:
                z = self.inference(batch[0])
                err.append(loss(batch[1], z))
                dh = (loss(batch[1], z + self.step) - loss(batch[1], z - self.step)) / (2 * self.step)
                dws[ind % 2] = list()
                dts[ind % 2] = list()
                for i in range(len(self.config) - 2, -1, -1):
                    dt = dh * self.derivative(activation, self.t[i])
                    dts[ind % 2].append(dt)
                    if ind != 0:
                        if dts[ind % 2][i] * dts[(ind + 1) % 2][i] > 0:
                            self.gamma[i] *= self.correction_plus
                        else:
                            self.gamma[i] *= self.correction_minus
                    if dt > 0:
                        self.b[i] -= self.gamma[i]
                    elif dt < 0:
                        self.b[i] += self.gamma[i]
                    dw = self.h[i + 1].transpose() @ dt
                    dws[ind % 2].append(dw)
                    if ind != 0:
                        if dws[ind % 2][i] * dws[(ind + 1) % 2][i] > 0:
                            self.delta[i] *= self.correction_plus
                        else:
                            self.delta[i] *= self.correction_minus
                    if dw > 0:
                        self.w[i] -= self.delta[i]
                    elif dw < 0:
                        self.w[i] += self.delta[i]
                    dh = dt @ self.w[i].transpose()
                ind += 1
            mean = np.mean(err)
            plot_err.append(mean)
            if (epoch + 1) % self.print_every == 0:
                print(f"Epoch {epoch + 1} loss: {mean}")
            self.save()
        if plot:
            fig, ax = plt.subplots()
            ax.plot(plot_err)
            plt.show()
