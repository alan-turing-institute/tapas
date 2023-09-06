"""
Toy example to showcase the reporting functions of TAPAS.

In TAPAS, an experiment is defined as running an attack on a  threat model
(representing the attacker knowledge) with ThreatModel.test. This method
outputs an AttackSummary object that contains all relevant information to
analyze the result (labels, predictions and scores, as well as metadata on
the experiment: the label of the dataset, generator, attack and target).
The results of one or more experiments (a collections of AttackSummaries)
are then fed to a tapas.report.Report object, that produces plots and/or
other files from the summaries.

In this example, we simulate the output of some attacks, and show how to use
tapas.report objects to produce human-readable plots from the output of an
experiment. 


"""

import numpy as np
import os

from tapas.report import (
    ALL_METRICS,
    MIAttackSummary,
    ROCReport,
    MIAttackReport,
    EffectiveEpsilonReport,
)


## 1. Common metrics and ROC curves.
# We simulate three attacks against three generators, with 100 samples each
# (i.e., 100 test synthetic datasets have been generated).

num_samples = 100

summaries = []

for generator_info in ["PlainsGen", "MountainGen", "IslandGen"]:
    for attack_name in ["CarrotAtk", "TomatoAtk", "BroccoliAtk"]:
        # Simulate the result of an attack (arbitrarily).
        noise_scale = 2 * np.random.random()
        fake_labels = np.random.randint(2, size=(num_samples,))
        fake_scores = fake_labels + np.random.normal(
            loc=0, scale=noise_scale, size=fake_labels.shape
        )
        summaries.append(
            # Typically, one will never need to create AttackSummaries manually,
            # as they are produced by ThreatModel.test.
            MIAttackSummary(
                fake_labels,
                fake_scores > 0.5,
                fake_scores,
                generator_info=generator_info,
                attack_info=attack_name,
                # Arbitrary names for the datasets and target.
                dataset_info="PrivateData",
                target_id="target_0",
            )
        )

# This is the folder where all the reports will be published.
report_folder = "reporting"

reports = [
    # The default summary: this outputs comparison plots for all (selected)
    # metrics over the different dimensions (generator/dataset/attack/target).
    # The plotted metrics are point estimates, and  the resulting plots tend
    # to be a bit messy.
    # NB: the default value of metrics is now DEFAULT_METRICS, which includes
    # a more restricted selection of metrics, leading to better readability.
    MIAttackReport(summaries, metrics=ALL_METRICS),
    # This summary can also provide estimated confidence intervals over point
    # estimates using bootstrapping of labels, predictions and scores. By default,
    # these are 95% confidence intervals, displayed as vertical lines around
    # the point estimates. 
    MIAttackReport(
        summaries, metrics=["accuracy", "privacy_gain", "auc"], num_bootstrap=100
    ),
    # The ROC curve is another important type of summary for privacy attacks,
    # as it gives a more granular view of vulnerability (i.e., is it possible
    # for an attacker to make highly accurate predictions for a small number
    # of outputs?).
    # By default, all experiments are displayed on the same plot. In this case,
    # it makes the result quite hard to interpret.
    ROCReport(summaries),
    # A solution is to disaggregate by a variable, i.e., group all curves with
    # the same value of a specific variable on the same plot. For instance, this
    # will produce one plot for each generator with the ROC curves of individual
    # attacks. This is useful to, e.g., select the best attack.
    ROCReport(summaries, disaggregate_by="generator"),
    # The following disaggregates by attack, comparing attack results for different
    # generators (which is less useful). In this case, we also change the label
    # used for the legend: each curve corresponds to a specific generator (the
    # default value is "attack", hence why we left it unchanged above). 
    ROCReport(summaries, disaggregate_by="attack", curve_label="generator"),
    # Additionally, we can "zoom in" on the bottom-left and top-right corners
    # of the ROC curve. This will produce two plots for each zoom < 1, and
    # allows to get a more granular idea of what happens in these critical
    # regions (high-precision attacks).
    ROCReport(summaries, disaggregate_by="generator", zooms=[0.25, 1]),
]

for i, report in enumerate(reports):
    # In this case, because we have several overlapping reports, we save the
    # outputs in different folders. In practice, you should save all reports
    # to the same folder (and not use the same object class twice, unless you
    # add a suffix (see below)).
    report.publish(os.path.join(report_folder, str(i)))


## 2. Effective Epsilon.
# This is a specific (niche?) type of report, which aims to determine whether a
# generator satisfied differential privacy, and for what value of epsilon.
# This requires a very large number of samples (as it relies on confidence
# intervals that need to be tight for good performance). For more details on
# this, please see the documentation at https://tapas-privacy.readthedocs.io/en/latest/evaluation.html.

# In this example, we assume we were able to run the generator 10000 times.
# This is typically impractical!
num_samples = 10000

# We generate fake scores by adding Laplace noise with scale 1. This is equivalent
# to a mechanism publishing M(D) = I{t \in D} + Lap(1), which is epsilon-DP for
# epsilon = 1. The effective epsilon we should expect is thus below 1.
fake_labels = np.random.randint(2, size=(num_samples,))
fake_scores = fake_labels + np.random.laplace(loc=0, scale=1, size=fake_labels.shape)
ee_summary = [
    MIAttackSummary(
        fake_labels,
        fake_scores > 0.5,
        fake_scores,
        generator_info="Laplace",
        attack_info="Optimal",
        dataset_info="PrivateData",
        target_id="target_0",
    )
]

reports = [
    # We report a ROC curved (zoomed in), on which we display the TP/FP and
    # TN/FN bounds entailed by epsilon-DP for epsilon = 1. This shows visually
    # that the mechanism satisfies these bounds (and matches them, in this case).
    ROCReport(ee_summary, zooms=[0.1, 1], eff_epsilon=1),
    # The effective epsilon report divides each summary in a validation part
    # (10%, by default) that is used to select the best attack and threshold,
    # and a test part (the remaining 90%) from which the effective epsilon is
    # estimated using Clopper-Pearson bounds.
    # The selection from validation data is done using a heuristic, either "cp"
    # (computing effective-epsilon from validation data for each attack and
    # threshold and selecting the maximum) by default, or "ratio" (which instead
    # used the TP/FP ratio). "cp" is the default, and tends to work the best,
    # although "ratio" can be useful when you suspect the mechanism does not
    # satisfy DP. The suffix argument is so that reports are saved to differently
    # named files (in practice, using EffectiveEpsilon(summaries) is enough).
    EffectiveEpsilonReport(ee_summary, heuristic='cp', suffix='cp'),
    EffectiveEpsilonReport(ee_summary, heuristic='ratio', suffix='ratio'),
    # You can also change the fraction of the experiment results used for validation
    # (i.e., attack+threshold selection). This will make the selection step more
    # robust, but will lead to a worse effective epsilon.
    EffectiveEpsilonReport(ee_summary, validation_split=0.5, suffix='cp_val')
]

# The above typically finds that, with 95% confidence, epsilon >= 0.9 (where the
# exact number varies with the randomness in this file). Since we know from the
# theory that epsilon <= 1, this suggests that the bound is tight.

for report in reports:
    report.publish(os.path.join(report_folder, 'effeps'))