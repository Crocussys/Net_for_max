from Matrix import Matrix
import json
import math
import random


def activation(x):
    return 1 / (1 + math.exp(-x))


class NeuralNet:
    def __init__(self, config: tuple = None, file_path: str = None):
        self.step = 0.00001
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
                self.w = Matrix(self.config[i - 1], self.config[i])
                for j in range(self.config[i - 1]):
                    for k in range(self.config[i]):
                        self.w.set(random.normalvariate(0, 1), j, k)
                self.b = Matrix(1, self.config[i])
                for k in range(self.config[i]):
                    self.b.set(random.normalvariate(0, 1), 1, k)
            self.save()

    def load(self):
        with open(self.file_path, 'r') as file:
            data = json.load(file)
        self.config = tuple(data["config"])
        self.w = data["w"]
        self.b = data["b"]

    def save(self):
        data = {"config": self.config, "w": self.w, "b": self.b}
        with open(self.file_path, 'w') as file:
            json.dump(data, file)

    def derivative(self, function, val):
        return (function(val + self.step) - function(val - self.step)) / (2 * self.step)

    def inference(self, x):
        self.t = list()
        self.h = [x]
        for i in range(len(self.config) - 1):
            self.t.append(self.h[i] @ self.w[i] + self.b[i])
            self.h.append(activation(self.t[i]))
        return self.h[-1]
