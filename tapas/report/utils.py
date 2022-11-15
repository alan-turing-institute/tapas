import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

from sklearn.metrics import roc_curve

# configurable axis ranges
axis_ranges = {
    "accuracy": (0, 1),
    "true_positive_rate": (0, 1),
    "false_positive_rate": (0, 1),
    "mia_advantage": (-0.2, 1.2),
    "privacy_gain": (-0.2, 1.2),
    "auc": (0, 1),
    "effective_epsilon": (0, 10),
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
        "accuracy", "true_positive_rate", "false_positive_rate", "mia_advantage", "privacy_gain", "auc".
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
                errorbar="ci",
            )
            axs[i].legend([], [], frameon=False)
            axs[i].set_ylabel(metric, fontsize=20)
            axs[i].set_xlabel("")
            axs[i].set_ylim(axis_ranges[metric])

        axs[-1].set_xlabel(f"{comparison_label}s".capitalize(), fontsize=20)

        handles, labels = axs[i].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center right",
            prop={"size": 20},
            bbox_to_anchor=(0.75, 0.25, 0.25, 0.3),
        )

        fig.suptitle(
            f"Comparison of {comparison_label}s and different targets"
            "\n"
            f"{fixed_pair_label[0]}: {pair_name[0]}, {fixed_pair_label[1]}: {pair_name[1]}",
            fontweight="bold",
            fontsize=24,
        )
        filename = f"{comparison_label}sComparison_Dataset{pair_name[0]}_Attack{pair_name[1]}.png"

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        plt.savefig(os.path.join(output_path, filename))

        plt.close(fig)


def plot_roc_curve(
    data,
    names,
    title,
    output_path,
    suffix="",
    eff_epsilon=None,
    zoom_in=1,
    low_corner=True,
):
    """
    Parameters
    ----------
    data: list of pairs (labels, scores), both np.arrays of same lengths
        The true labels and the scores of each attack.
    names: list of str of the same length
        The label for each curve.
    title: str
        Title to display on the figure.
    output_path: str
        Path to the folder where the ROC curve should be saved.
    eff_epsilon: positive float, or None
        If not None, the value of the effective epsilon for this ROC curve,
        for which the TP/FP and (1-FP)/(1-TP) curves are plotted.
    zoom_in: float, default 1
        Maximum value of TP and FP shown on the plot. The default of 1 shows
        the full ROC curve, but this can be used to "zoom in" to the TPR at
        low FPR, an important quantity for privacy analysis.
    low_corner: bool, default True
        Whether to zoom in near (0,0) (True), or (1,1) (False).

    """
    set_style()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.subplots()

    # Plot the "baseline".
    decorum_color = (0.7, 0.7, 0.7)

    ax.plot([0, 1], [0, 1], "--", color=decorum_color)

    if eff_epsilon is not None:
        assert eff_epsilon > 0, "eff_epsilon must be positive."
        slope = np.exp(eff_epsilon)
        tp_inter = slope / (slope + 1)
        fp_inter = 1 / (slope + 1)
        ax.plot([0, fp_inter], [0, tp_inter], "--", color=decorum_color)
        ax.plot([fp_inter, 1], [tp_inter, 1], "--", color=decorum_color)

    for (labels, scores), name in zip(data, names):
        fpr, tpr, thresholds = roc_curve(labels, scores)
        ax.plot(fpr, tpr, label=name)

    ax.legend(loc="lower right", fontsize=20)

    # We add a small margin to protect [0,1].
    margin = 0.01
    if low_corner:
        ax.set_xlim([0 - margin * zoom_in, zoom_in])
        ax.set_ylim([0, (1 + margin) * zoom_in])
    else:
        ax.set_xlim([1 - zoom_in, 1])
        ax.set_ylim([1 - zoom_in, 1 + margin * zoom_in])
    ax.set_xlabel("False Positive Rate", fontsize=20)
    ax.set_ylabel("True Positive Rate", fontsize=20)

    if title:
        fig.suptitle(title, fontweight="bold", fontsize=24)

    filename = f"ROC_curve{suffix}.png"
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
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "savefig.dpi": 75,
            "figure.autolayout": False,
            "figure.figsize": (12, 10),
            "figure.titlesize": 18,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "legend.fontsize": 14,
        }
    )
