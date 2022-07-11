import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

# configurable axis ranges
axis_ranges = {
    "accuracy": (0, 1),
    "true_positive_rate": (0, 1),
    "false_positive_rate": (0, 1),
    "mia_advantage": (-0.2, 1.2),
    "privacy_gain": (-0.2, 1.2),
}
color_pal = sns.color_palette("colorblind", 10)


def metric_comparison_plots(
    data, comparison_label, fixed_pair_label, metrics, marker_label, output_path
):

    """
    For a fixed pair of datasets-attacks-generators-target available in the data make a figure comparing
        performance between metrics. Options configure which dimension to compare against. Figures are saved to disk.

    Parameters
    ----------
    data: dataframe
        Input dataframe from the MIAttackReport class
    comparison_label: str
        Name of column that will be use as X axis
    fixed_pair_columns: list[str]
             Columns in dataframe to fix (groupby) for a given figure in order to make meaningful comparisons. It can be any pair
    metrics:  list[str]
        List of metrics to be used in the report, these can be any of the following:
        "accuracy", "true_positive_rate", "false_positive_rate", "mia_advantage", "privacy_gain".
    marker_label: str
        Column in dataframe that be used to as marker in a point plot comparison. It can be either: 'generator',
    'attack' or 'target_id'.
    output_path: str
        Path where the figure is to be saved.

    Returns
    -------
    None

    """
    set_style()

    for pair_name, pair in data.groupby(fixed_pair_label):
        fig, axs = plt.subplots(len(metrics), sharex=True)

        for i, metric in enumerate(metrics):
            sns.pointplot(
                data=pair,
                y=metric,
                x=comparison_label,
                hue=marker_label,
                order=np.unique(pair[comparison_label]),
                ax=axs[i],
                dodge=True,
                errwidth=1,
                linestyles="",
            )
            axs[i].legend([], [], frameon=False)
            axs[i].set_ylabel(metric)
            axs[i].set_xlabel("")
            axs[i].set_ylim(axis_ranges[metric])

        axs[-1].set_xlabel(f"{comparison_label}s")

        handles, labels = axs[i].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", prop={"size": 12})

        fig.suptitle(
            f"Comparison of {comparison_label}s and different targets"
            "\n"
            f"{fixed_pair_label[0]}: {pair_name[0]}, {fixed_pair_label[1]}: {pair_name[1]}",
            fontweight="bold",
        )
        filename = f"{comparison_label}sComparison_Dataset{pair_name[0]}_Attack{pair_name[1]}.png"

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        plt.savefig(os.path.join(output_path, filename))

        plt.close(fig)


def set_style():

    sns.set_palette(color_pal)
    sns.set_style(
        "whitegrid",
        {
            "axes.spines.right": True,
            "axes.spines.top": True,
            "axes.edgecolor": "k",
            "xtick.color": "k",
            "xtick.rotation": 45,
            "ytick.color": "k",
            "font.family": "sans-serif",
            "font.sans-serif": "Tahoma",
            "text.usetex": True,
        },
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": "Tahoma",
            "font.size": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "savefig.dpi": 75,
            "figure.autolayout": False,
            "figure.figsize": (12, 10),
            "figure.titlesize": 18,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "legend.fontsize": 14,
        }
    )
