import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np


def metric_comparison_plots(data, comparison_label, fixed_pair_label, metrics, marker_label, output_path):

    for pair_name, pair in data.groupby(fixed_pair_label):
        fig, axs = plt.subplots(len(metrics), figsize=(10, 10), sharex=True)

        for i, metric in enumerate(metrics):
            sns.pointplot(data=pair, y=metric,
                          x=comparison_label, hue=marker_label,
                          order=np.unique(pair[comparison_label]),
                          ax=axs[i], dodge=True,
                          errwidth=1, linestyles='')
            axs[i].legend([], [], frameon=False)
            axs[i].set_ylabel(metric)
            axs[i].set_xlabel('')

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
