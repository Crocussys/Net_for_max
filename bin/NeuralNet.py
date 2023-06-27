from bin.Matrix import *
from bin.functions import sigmoid
import json
import random


class NeuralNet:
    def __init__(self, config: tuple = None, file_path: str = None):
        self.print_every = 1
        self._w = list()
        self._b = list()
        self._t = list()
        self._h = list()
        self.__file_path = file_path
        self._config = config
        self.activation_func = sigmoid
        if config is None:
            if file_path is None:
                raise ValueError
            self.load()
        else:
            if file_path is None:
                self.__file_path = "network.json"
            for i in range(1, len(config)):
                self._w.append(Matrix(self._config[i - 1], self._config[i]))
                for j in range(self._config[i - 1]):
                    for k in range(self._config[i]):
                        self._w[i - 1][j][k] = random.normalvariate(0, 1)
                self._b.append(Vector(self._config[i]))
                for k in range(self._config[i]):
                    self._b[i - 1][k] = random.normalvariate(0, 1)
            self.save()

    def load(self):
        with open(self.__file_path, 'r') as file:
            data = json.load(file)
        self._config = tuple(data["config"])
        self._w = list()
        for w_mtrx in data["w"]:
            self._w.append(Matrix().from_list(w_mtrx))
        self._b = list()
        for b_vect in data["b"]:
            self._b.append(Vector().from_list(b_vect))

    def save(self):
        data = {"config": self._config}
        temp = list()
        for w_mtrx in self._w:
            temp.append(w_mtrx.in_list())
        data.update({"w": temp})
        temp = list()
        for b_vect in self._b:
            temp.append(b_vect.in_list())
        data.update({"b": temp})
        with open(self.__file_path, 'w') as file:
            json.dump(data, file)

    def get_config(self):
        return self._config

    def inference(self, x):
        self._t = list()
        self._h = [x]
        for i in range(len(self._config) - 1):
            self._t.append(self._h[i] @ self._w[i] + self._b[i])
            self._h.append(self._t[i].in_func(self.activation_func, "x"))
        return self._h[-1]
