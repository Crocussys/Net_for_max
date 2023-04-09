import matplotlib.pyplot as plt
import numpy as np
import json


def activation(x):
    return 1 / (1 + np.exp(-x))


def loss(x, y):
    return np.sum((x - y) ** 2) / len(x)


class NeuralNet:
    def __init__(self, config: tuple = None, file_path: str = None):
        self.step = 0.00001
        self.learn_rate = 0.01
        self.epochs = 1000
        self.print_every = 100
        self.w = list()
        self.b = list()
        self.t = list()
        self.h = list()
        self.file_path = file_path
        self.config = config
        if config is None:
            if file_path is None:
                raise ValueError
            self.load()
        else:
            if file_path is None:
                self.file_path = "network.json"
            for i in range(1, len(config)):
                rng = np.random.default_rng()
                self.w.append(rng.standard_normal(size=(self.config[i - 1], self.config[i]), dtype=np.float64))
                self.b.append(rng.standard_normal(size=(self.config[i]), dtype=np.float64))
            self.save()

    def load(self):
        with open(self.file_path, 'r') as file:
            data = json.load(file)
        self.w = list()
        self.b = list()
        self.config = tuple(data["config"])
        for w in data["w"]:
            self.w.append(np.array(w, dtype=np.float64))
        for b in data["b"]:
            self.b.append(np.array(b, dtype=np.float64))

    def save(self):
        ws = list()
        for w in self.w:
            ws.append(w.tolist())
        bs = list()
        for b in self.b:
            bs.append(b.tolist())
        data = {"config": self.config, "w": ws, "b": bs}
        with open(self.file_path, 'w') as file:
            json.dump(data, file)

    def inference(self, x):
        self.t = list()
        self.h = [x]
        for i in range(len(self.config) - 1):
            self.t.append(self.h[i] @ self.w[i] + self.b[i])
            self.h.append(activation(self.t[i]))
        return self.h[-1]

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
