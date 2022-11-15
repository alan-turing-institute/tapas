==========
Evaluation
==========

The module ``tapas.report`` provides tools to analyse the outcome of attacks, and in particular compare different attacks and threat models.
The outcome of an attack is obtained from ``ThreatModel.test(attack)``, which returns an ``AttackSummary`` object.
This object contains information about how the attack performed: typically, the outcome for a number of simulations.

``Report`` objects aggregate ``AttackSummary`` objects from different attacks to produce a human-readable output.
These objects provide a ``.publish(filepath)`` method, which publishes a "report" of these attacks, saved as one or more files in the folder indicated by ``filepath``. ``TAPAS`` implements different reports.



Classification Metrics
----------------------

``BinaryLabelAttackReport`` and its subclasses ``MIAttackReport``/``AIAttackReport`` aggregate summaries resulting from membership- or attribute-inference attacks by computing a range of classification metrics on each summary. The ``.publish`` method generates a range of plots, showing all metrics across different datasets, attacks, and generators.

The metrics are extracted from the ``AttackSummary``, and thus all metrics implemented in the latter transfer to the ``Report``.
For binary classification tasks (MIA/Binary AIA), the following metrics are generated:

- ``accuracy``: the success rate of the attack.
- ``true_positive_rate``: the probability that the attack makes a correct guess for a positive record (label 1).
- ``false_positive_rate``: the probability that the attack makes a correct guess for a negative record (label 0).
- ``mia_advantage``: advantage of the attack over a random attacker, defined by Stadler et al. [1]_.
- ``privacy_gain``: the privacy gain of the generator over publishing the data, defined by Stadler et al. [1]_
- ``auc``: the area under the ROC curve of the scores used by the attack.
- ``effective_epsilon``: the estimated :math:`\varepsilon` of the attack (see below).

``BinaryLabelAttackReport`` objects have an optional parameter:

- ``metrics`` (default ``None``): the list of metrics to report in plots. By default, all the metrics are used.

The child classes ``MIAttackReport`` and ``AIAttackReport`` expect the attack summary to have additional attributes describing the attack and threat model (specifically, the dataset, generator, target and attack labels) that are used when labelling the plots.

The plots produced by ``BinaryLabelAttackReport.publish`` display all specified metrics, each in a separate subplot, and disaggregate the result by four dimensions: the dataset the attack is run on, the generator attacked, the target of the attack, and the attack used. By default, plots are produced that compare across each dimension, using the ``.compare`` method.
This method generates plots for all values of two specific columns (``fixed_pair_columns``), where the comparison is performed according to another column, ``comparison_column``.
The choice of disaggregation (dataset/generator/target/attack) is inspired by the work of Stadler et al., and intends to cover a range of common use cases. Other comparions (e.g., across threat models) must be implemented manually.

.. TODO: this seems like something to look into. 


ROC curve
---------

``ROCReport`` aggregates the summaries by plotting a ROC (receiver operating characteristic) curve for each attack. For an attack defined by its score :math:`s` and a variable threshold :math:`\tau`, :math:`\mathcal{A}: D \mapsto s(D) \geq \tau`, the ROC curve plots the true positive rate against the false positive rate for a range of threshold values, :math:`C = \left(FP_{\tau_i}, TP_{\tau_i}\right)_{i=1,\dots,k}`.
The ``publish`` method produces an image combining ROC curves for each summary.
Although these summaries can correspond to anything, it is recommended to have summaries produced by different attacks under the same threat model.



Effective epsilon
-----------------

Let :math:`\mathcal{M}` be a mechanism satisfying :math:`\varepsilon`-differential privacy. A classical result is that an attack :math:`\mathcal{A}` aiming to differentiate between two neighbouring datasets :math:`D \sim D'` from the output of :math:`\mathcal{M}` must satisfy the following bound relating its true and false positive rates:

.. math::
	\max\left(\frac{TP_\mathcal{A}}{FP_\mathcal{A}}, \frac{1-FP_\mathcal{A}}{1-TP_\mathcal{A}}\right) \leq e^\varepsilon

This bound tells us that no nontrivial attack can achieve a true positive rate of 1, or a true negative rate of 0.
This also gives us a metric for attack success, in the value of :math:`\varepsilon` for which this bound holds true.
We thus define the effective epsilon :math:`\varepsilon^\text{eff}` of an attack (against a specific mechanism :math:`\mathcal{M}` and for a specific pair of datasets) as:

.. math::
	\varepsilon^\text{eff}_\mathcal{M}(\mathcal{A}) = \frac{TP_\mathcal{A}}{FP_\mathcal{A}}

Importantly, this is a very noisy metric: estimating the false positive rate from a finite sample can lead to underestimation, and crucially estimating :math:`FP_\mathcal{A} = 0`, in which case the metric degenerates. For this reason, one should instead use a statistically significant lower bound for TP and upper bound for FP, leading to a trustworthy estimate. This methodology was introduced by Jagielski et al. [2]_, and uses Clopper-Pearson bounds for the confidence interval of a binomial distribution.

Since ``TAPAS`` aims at evaluating the robustness of a synthetic data generation mechanism :math:`\mathcal{M}`, a more interesting metric is the effective epsilon of *the mechanism*:

.. math::
	\varepsilon^\text{eff}(\mathcal{M}) = \sup_{\mathcal{A}} \varepsilon^\text{eff}_\mathcal{M}(\mathcal{A})

Iterating over all possible attacks is of course impossible. Instead, ``TAPAS`` implements and applies a finite number of attacks :math:`\left(\mathcal{A}_{1, \tau_1}, \dots, \mathcal{A}_{n, \tau_n}\right)` with variable thresholds :math:`\left(\tau_1, \dots, \tau_n\right)`. The effective epsilon is then estimated as:

.. math::
	\widehat{\varepsilon^\text{eff}}(\mathcal{M}) = \max_{i, \tau} \varepsilon^\text{eff}_\mathcal{M}(\mathcal{A}_{i,\tau})

``EffectiveEpsilonReport`` implements this idea, by performing the following two-step procedure:

1. Select a candidate attack :math:`\mathcal{A}_i` and threshold :math:`\tau` that will likely lead to high effective epsilon, from a set of attack outcomes :math:`S_1`. This is done by estimating the effective epsilon with a low confidence level on :math:`S_1`.
2. Estimate effective epsilon with Clopper-Pearson bounds for this attack from a disjoint set of attack outcomes :math:`S_2`. Repeat this step with different levels of confidence :math:`\left(\gamma_1, \dots, \gamma_l\right)` and report the estimatee obtained for each level.

The ``publish`` method produces a ``Pandas`` dataframe containing the estimate obtained with the candidate attack for each level of confidence. The results are returned and also saved to disk as a CSV file.

The ``EffectiveEpsilonReport`` constructor has two parameters:

- ``validation_split``: the *fraction* of all attack outcomes to use in the first step ("validation"). The rest is used for the second step ("test"). The default value is 0.1, as the second part of the analysis is very sample intensive.
- ``confidence_levels``: a tuple of confidence levels to report. You can add as many as are needed, but for most use cases the default value  of ``(0.9, 0.95, 0.99)`` should suffice.

A few important notes:

- This analysis assumes that all attacks are performed in the exact same setup (same ``ThreatModel``).
- While this analysis can in theory be applied to any type of auxiliary knowledge, one of its key applications is testing whether the DP guarantees of a mechanism :math:`\mathcal{M}` are correct. In this setup, it is recommended to use ``ExactDataKnowledge`` as auxiliary knowledge, as it is closest to the setup of the bound described earlier.
- The effective epsilon estimated by this procedure is *very conservative*, and requires many samples for nontrivial results. However, the results obtained are statistically significant.


References
----------

.. [1] Stadler, T., Oprisanu, B. and Troncoso, C., 2021. Synthetic data–anonymisation groundhog day. arXiv preprint arXiv:2011.07018.
.. [2] Jagielski, M., Ullman, J. and Oprea, A., 2020. Auditing differentially private machine learning: How private is private sgd?. Advances in Neural Information Processing Systems, 33, pp.22205-22216.
