from NeuralNet import NeuralNet
import matplotlib.pyplot as plt
import numpy as np
import json
import csv


def in_csv(path: str, split: int):
    data = list()
    with open(path, 'r') as file:
        reader = list(csv.reader(file, delimiter=';'))
        for i in range(len(reader)):
            line = reader[i]
            size = len(line)
            data.append([np.empty(split, dtype=np.float64), np.empty(size - split, dtype=np.float64)])
            for j in range(size):
                if j < split:
                    data[i][0][j] = line[j].replace(',', '.')
                else:
                    data[i][1][j - split] = line[j].replace(',', '.')
    return data


def inference(input_file: str, output_file: str = None):
    if output_file is None:
        output_file = "output.csv"
    output_dim = Network.config[-1]
    data = in_csv(input_file, Network.config[0])
    net_outs = np.empty((len(data), output_dim))
    true_outs = np.empty((len(data), output_dim))
    i = 0
    for inp in data:
        net_out = Network.inference(inp[0])
        print(net_out)
        net_outs[i] = net_out
        true_outs[i] = inp[1]
        i += 1
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(net_outs)
    if config_data["plot"]:
        if output_dim > 1:
            fig, ax = plt.subplots(output_dim)
            net_outs = net_outs.transpose()
            true_outs = true_outs.transpose()
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


if __name__ == '__main__':
    with open('config.json', 'r') as config_file:
        config_data = json.load(config_file)
    if config_data["create"]:
        Network = NeuralNet(tuple(config_data["config_network"]), file_path=f"{config_data['path']}\\network.json")
    else:
        Network = NeuralNet(file_path=f"{config_data['path']}\\network.json")
    Network.epochs = config_data["epochs"]
    Network.print_every = config_data["print_every"]
    Network.learn_rate = config_data["learn_rate"]
    if config_data["learn"]:
        Network.learning(in_csv(f"{config_data['path']}\\learning.csv", Network.config[0]), config_data["plot"])
    if config_data["inference"]:
        inference(f"{config_data['path']}\\input.csv", f"{config_data['path']}\\output.csv")
