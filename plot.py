"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf
import numpy as np
import argparse
import os
import json
import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--title', type=str)
    parser.add_argument('--xlabel', type=str)
    parser.add_argument('--ylabel', type=str)
    parser.add_argument('--input_patterns', nargs='+', type=str, default=[])
    parser.add_argument('--input_names', nargs='+', type=str, default=[])
    parser.add_argument('--bars', nargs='+', type=float, default=[])
    parser.add_argument('--bar_names', nargs='+', type=str, default=[])
    args = parser.parse_args()

    # create the directory to house the plot
    tf.io.gfile.makedirs(os.path.dirname(args.output_file))

    # load the y values from json files for each pattern
    input_lengths = dict()
    input_iterations = dict()
    data = dict()
    max_iteration = 0
    min_iteration = 0

    for name, pattern in zip(args.input_names, args.input_patterns):
        input_lengths[name] = 0
        data[name] = []

        for file_name in tf.io.gfile.glob(pattern):
            with tf.io.gfile.GFile(file_name, "r") as f:
                values = np.array(json.load(f))
                max_iteration = max(max_iteration, values[-1, 1])
                min_iteration = min(min_iteration, values[0, 1])
                data[name].append(values[:, 2])

                if values.shape[0] > input_lengths[name]:
                    input_lengths[name] = values.shape[0]
                    input_iterations[name] = values[:, 1]

    # load the y values from json files for each pattern
    plt.clf()
    ax = plt.subplot(111)

    for name, pattern in zip(args.input_names, args.input_patterns):
        for i in range(len(data[name])):
            values = data[name][i]

            data[name][i] = np.concatenate([
                values,
                np.full([input_lengths[name] - values.shape[0]], values[-1])], 0)

        data[name] = np.stack(data[name], axis=0)
        mean = np.mean(data[name], axis=0)
        std = np.std(data[name], axis=0)

        # plot the curve using a random color
        color = np.random.uniform(low=(0.0, 0.0, 0.0), high=(1.0, 1.0, 1.0))
        ax.plot(input_iterations[name], mean, color=color, label=name)
        ax.fill_between(
            input_iterations[name],
            mean - std,
            mean + std,
            color=np.concatenate([color, [0.2]], 0))

    # plot horizontal bars using random color
    for name, value in zip(args.bar_names, args.bars):
        color = np.random.uniform(low=(0.0, 0.0, 0.0), high=(1.0, 1.0, 1.0))
        ax.plot(
            [min_iteration, max_iteration],
            [value, value],
            color=color,
            label=name,
            linestyle=":")

    # clean up the plot a bit
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)

    ax.set_title(args.title)
    ax.legend()
    plt.savefig(args.output_file)
