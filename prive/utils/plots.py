import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from palettable import cartocolors


def metric_comparison_plots(data, comparison_label, fixed_pair_label, metrics, marker_label, output_path):
    set_style()
    axis_ranges = {"accuracy": (0, 1), "true_positive_rate": (0, 1), "false_positive_rate": (0, 1),
                   "mia_advantage": (-0.2, 1.2),
                   "privacy_gain": (-0.2, 1.2)}

    for pair_name, pair in data.groupby(fixed_pair_label):
        fig, axs = plt.subplots(len(metrics), sharex=True)

        for i, metric in enumerate(metrics):
            sns.pointplot(data=pair, y=metric,
                          x=comparison_label, hue=marker_label,
                          order=np.unique(pair[comparison_label]),
                          ax=axs[i], dodge=True,
                          errwidth=1, linestyles='')
            axs[i].legend([], [], frameon=False)
            axs[i].set_ylabel(metric)
            axs[i].set_xlabel('')
            axs[i].set_ylim(axis_ranges[metric])

        axs[-1].set_xlabel(f'{comparison_label}s')

        handles, labels = axs[i].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')

        fig.suptitle(
            f'Comparison of {comparison_label}s and different targets'
            '\n'
            f'{fixed_pair_label[0]}: {pair_name[0]}, {fixed_pair_label[1]}: {pair_name[1]}', fontweight='bold')
        filename = f'{comparison_label}sComparison_Dataset{pair_name[0]}_Attack{pair_name[1]}.png'

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        plt.savefig(os.path.join(output_path, filename))

        plt.close(fig)


def set_style():
    #colours = cartocolors.qualitative.Safe_8.hex_colors
    #cpalette = sns.color_palette(colours)
    color_pal = sns.color_palette("colorblind", 10)

    sns.set_palette(color_pal)
    sns.set_style('whitegrid', {'axes.spines.right': True,
                            'axes.spines.top': True,
                            'axes.edgecolor': 'k',
                            'xtick.color': 'k',
                            'xtick.rotation': 45,
                            'ytick.color': 'k',
                            'font.family': 'sans-serif',
                            'font.sans-serif': 'Tahoma',
                            'text.usetex': True})

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': 'Tahoma',
        'font.size': 10,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'savefig.dpi': 75,
        'figure.autolayout': False,
        'figure.figsize': (10, 10),
        'figure.titlesize': 18,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'legend.fontsize': 14,
    })
