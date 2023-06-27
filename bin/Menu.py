from bin.training_methods import *
import json
import csv


def in_csv(path: str, split: int, delimiter=';'):
    data = list()
    with open(path, 'r') as file:
        reader = list(csv.reader(file, delimiter=delimiter))
        for i in range(len(reader)):
            line = reader[i]
            size = len(line)
            data.append([Vector(split), Vector(size - split)])
            for j in range(size):
                if j < split:
                    data[i][0][j] = float(line[j].replace(',', '.'))
                else:
                    data[i][1][j - split] = float(line[j].replace(',', '.'))
    return data


class Menu:
    def __init__(self, lang="RU"):
        with open("bin\\lang.json", 'r', encoding="UTF-8") as lang_file:
            self.__lang = json.load(lang_file)[lang]
        with open("config.json", 'r', encoding="UTF-8") as config_file:
            self.__config = json.load(config_file)
        self.__path = self.__config["path"]

    def question(self, dat, zero=None):
        print(dat["question"])
        answers = dat["answers"]
        count_answers = len(answers)
        for i in range(count_answers):
            print(f"{i + 1}. {answers[i]}")
        if zero is not None:
            print(f"0. {zero}")
        while True:
            inp = input("> ")
            try:
                inp = int(inp)
            except ValueError:
                print(self.__lang["errors"]["NOT_NUMBER"])
                continue
            if zero is not None:
                if inp < 0 or inp > count_answers:
                    print(self.__lang["errors"]["NOT_ANSWER"])
                    continue
            else:
                if inp <= 0 or inp > count_answers:
                    print(self.__lang["errors"]["NOT_ANSWER"])
                    continue
            return inp

    def config(self):
        print(self.__lang["messages"]["config"])
        while True:
            inp = input("> ").split(" ")
            size = len(inp)
            if size < 2:
                print(self.__lang["errors"]["NOT_FORMAT"])
                continue
            error_flag = False
            for i in range(size):
                if error_flag:
                    break
                try:
                    inp[i] = int(inp[i])
                except ValueError:
                    print(self.__lang["errors"]["NOT_FORMAT"])
                    error_flag = True
                    continue
                if inp[i] <= 0:
                    print(self.__lang["errors"]["NOT_FORMAT"])
                    error_flag = True
                    continue
            if error_flag:
                continue
            return tuple(inp)

    def start(self):
        print(self.__lang["messages"]["hi"])
        print(f'{self.__lang["messages"]["path"]}: {self.__path}')
        create = self.question(self.__lang["create"])
        if create == 1:
            config = self.config()
        else:
            config = None
        learn = self.question(self.__lang["learn"])
        if learn == 1:
            training_method = self.question(self.__lang["training_method"])
        else:
            training_method = None
        inference = self.question(self.__lang["inference"])
        if learn == 1 or inference == 1:
            plot = self.question(self.__lang["plot"])
        else:
            plot = None

        network = None
        if learn == 2:
            if create == 2:
                network = NeuralNet(file_path=f"{self.__path}\\network.json")
            elif create == 1:
                network = NeuralNet(config, file_path=f"{self.__path}\\network.json")
        elif learn == 1:
            if training_method == 1:
                if create == 2:
                    network = BackProp(file_path=f"{self.__path}\\network.json")
                elif create == 1:
                    network = BackProp(config, file_path=f"{self.__path}\\network.json")
            elif training_method == 2:
                if create == 2:
                    network = ResilientProp(file_path=f"{self.__path}\\network.json")
                elif create == 1:
                    network = ResilientProp(config, file_path=f"{self.__path}\\network.json")
            if plot == 1:
                network.learning(in_csv(f"{self.__path}\\learning.csv", network.get_config()[0]), True)
            elif plot == 2:
                network.learning(in_csv(f"{self.__path}\\learning.csv", network.get_config()[0]), False)

        if inference == 1:
            output_dim = network.get_config()[-1]
            data = in_csv(f"{self.__path}\\input.csv", network.get_config()[0])
            net_outs = list()
            true_outs = list()
            for inp in data:
                net_out = network.inference(inp[0])
                print(net_out)
                net_outs.append(net_out)
                true_outs.append(inp[1])
            with open(f"{self.__path}\\output.csv", 'w', newline='') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
                writer.writerows(net_outs)
            if plot == 1:
                if output_dim > 1:
                    fig, ax = plt.subplots(output_dim)
                    for i in range(output_dim):
                        ax[i].plot(net_outs[i], label="Нейросеть")
                        ax[i].plot(true_outs[i], label="Истинные")
                    plt.legend()
                    plt.show()
                else:
                    fig, ax = plt.subplots()
                    ax.plot(net_outs, label="Нейросеть")
                    ax.plot(true_outs, label="Истинные")
                    plt.legend()
                    plt.show()
