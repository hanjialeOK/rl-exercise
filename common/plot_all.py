import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import argparse
import matplotlib.patches as mpatches
from matplotlib.ticker import EngFormatter

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
units = dict()


def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition="Condition", smooth=1, ax=None, **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])  # (200, 1)
            x = np.squeeze(x)  # (200,)
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    # sns.set(style="darkgrid", palette="deep", font_scale=1.5)
    # ax.margins(0.05)
    sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", legend=False,
               condition=condition, ci=50, ax=ax, linewidth=2.5, **kwargs)
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from
    tsplot to lineplot replacing L29 with:
        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)
    Changes the colorscheme and the default legend style, though.
    """

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    yscale = np.max(np.asarray(data[value])) > 5e4
    if yscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax.tight_layout(pad=0.5)


"""
Iteraion    Value   Unit    Condition
..          ..      0       dqn
..          ..      1       dqn
..          ..      2       dqn
..          ..      3       dqn
..          ..      4       dqn
..          ..      0       ddqn
..          ..      1       ddqn
..          ..      2       ddqn
..          ..      3       ddqn
..          ..      4       ddqn
..          ..      0       duel
..          ..      1       duel
..          ..      2       duel
..          ..      3       duel
..          ..      4       duel
"""


def get_datasets(logdir, legend=None, tag=None, data_file='progress.txt'):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.
    Assumes that any file "progress.txt" is a valid hit.
    """
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if ('progress.txt' in files) or ('progress.csv' in files):
            exp_name = None
            try:
                config_path = open(os.path.join(root, 'config.json'))
                config = json.load(config_path)
                if tag in config:
                    exp_name = config[tag]
            except:
                print('No file named config.json')
            condition = legend or exp_name
            if condition not in units:
                units[condition] = 0
            unit = units[condition]
            units[condition] += 1

            try:
                if 'progress.txt' in files:
                    exp_data = pd.read_table(
                        os.path.join(root, 'progress.txt'))
                elif 'progress.csv' in files:
                    exp_data = pd.read_csv(os.path.join(root, 'progress.csv'))
            except:
                print('Could not read from %s' %
                      os.path.join(root, data_file))
                continue
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition', condition)
            datasets.append(exp_data)
    return datasets, condition


def main(args):
    envs = ['Ant', 'HalfCheetah', 'Hopper', 'Humanoid', 'HumanoidStandup',
            'InvertedDoublePendulum', 'InvertedPendulum', 'Reacher', 'Swimmer', 'Walker2d']
    algs = ['PPO-master-joschu', 'PPO-master-lrdecay', 'PPO-jqueeney-spinnup']
    legends = ['PPO-master-joschu',
               'PPO-master-lrdecay', 'PPO-jqueeney-spinnup']
    version = 'v2'

    nsize = (2, 5)

    sns.set(style="darkgrid", palette="deep", font_scale=1.8)
    fig, axis = plt.subplots(nrows=nsize[0], ncols=nsize[1],
                             figsize=(6.4*nsize[1], 4.8*nsize[0]*1.2))

    for i in range(len(envs)):
        env_name = envs[i] + '-' + version
        base_dir = os.path.join(args.logdir, env_name)
        if not os.path.exists(base_dir):
            raise ValueError(f'No such basedir: {base_dir}')

        # Check whether the logdirs are valid.
        logdirs = []
        for alg in algs:
            alg_dir = os.path.join(base_dir, alg)
            if os.path.isdir(alg_dir):
                logdirs += [alg_dir]
            else:
                raise ValueError(f'No such algdir: {alg_dir}')

        # legend and logdir must be corresponding.
        if legends:
            assert len(legends) == len(logdirs)

        print('Getting data from...\n' + '='*DIV_LINE_WIDTH + '\n')

        data = []
        global units
        units = dict()
        for logdir, legend in zip(logdirs, legends):
            datasets, cond = get_datasets(
                logdir=logdir, legend=legend, tag=args.tag, data_file=args.file)
            data += datasets
            print(f'{logdir} -> {cond}, {units[cond]}')

        print('\n' + '='*DIV_LINE_WIDTH)
        print('Plotting...')

        # Choose which subplot to plot
        ax = axis[i // nsize[1], i % nsize[1]]

        plot_data(data, xaxis=args.xaxis, value=args.value, condition='Condition',
                  smooth=args.smooth, ax=ax)

        ax.set_title(env_name)
        ax.set_xlabel('')
        ax.set_ylabel('')

        # if i // nsize[1] == 1:
        #     ax.set_xlabel('Steps(M)', labelpad=10)
        # else:
        #     ax.set_xlabel('')

        # if i % nsize[1] == 0:
        #     ax.set_ylabel('Average Episode Reward')
        #     ax.set_ylabel('')
        # else:
        #     ax.set_ylabel('')

        ax.xaxis.set_major_formatter(EngFormatter())

        # margin = 1e6 * 0.05
        # ax.set_xlim(0, 1e6)
        # ax.set_xlim(0-margin, 1e6+margin)
        # ax.margins(0.05)
        ax.autoscale()
        ax.set_box_aspect(4.8/6.4)

    plt.tight_layout(pad=0.5)
    lines = fig.axes[-1].get_lines()
    # for line in lines:
    #     line.set_linewidth(2.0)
    handles = [mpatches.Patch(color=line.get_c(), label=legend)
               for (line, legend) in zip(lines, legends)]
    fig.legend(handles=handles, loc='lower center', prop={'size': 25},
               ncol=len(algs), handlelength=1, borderaxespad=0.)
    # fig.legend(handles=lines, labels=legends, loc='lower center', prop={'size': 20},
    #            ncol=len(algs), handlelength=2, borderaxespad=0.)
    plt.subplots_adjust(bottom=0.1)
    plt.show()
    plt.savefig('all.pdf')
    plt.savefig('all.svg')
    print(f"Saved into {os.path.abspath('all.pdf')}")


"""
Usage:
    python rl-exercise/lib/utils/plot.py --logdir data/bench_dqn
    data/bench_ddqn data/bench_dueling --xaxis Iteration --value Value
    --smooth 8 --tag exp_name --legend my_dqn my_ddqn my_dueling
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True, type=str)
    parser.add_argument('--legend', '-l', default=None, nargs='*')
    parser.add_argument('--xaxis', '-x', default='Step')
    parser.add_argument('--value', '-y', default='AvgEpRet')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--tag', type=str, default='exp_name')
    parser.add_argument('--file', type=str, default='progress.txt')
    parser.add_argument('--name', type=str, default='exp')
    args = parser.parse_args()
    main(args)
