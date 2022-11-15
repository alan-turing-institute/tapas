==================
Library of Attacks
==================

Adversarial approaches for privacy require auditors to run a large range of diverse attacks, in order to test for as many potential vulnerabilities as possible.
For this reason, ``TAPAS`` implements a large range of attacks.
While some of these attacks are technically involved, many are straightforward and are intended mostly as safety checks.
We here present the different attacks implemented in ``TAPAS``, grouped by theme. For each attack, we specify the additional parameters it requires, and the attack models (see `Modelling Threats <modelling-threats.rst>`) it applies to.
``TAPAS`` attacks inherit from the ``tapas.attacks.Attack`` abstract class (see `Implementing Attacks <implementing-attacks.rst>` for details).
Note that the constructor of *all* the attacks described below allows for an optional ``label`` parameters, which we exclude from the descriptions for the sake of concision.

*Notations*: we denote by :math:`D^{(r)}` the real, private dataset, and :math:`D^{(s)}` the synthetic dataset obtained with the generation method :math:`\mathcal{G}`. For targeted attacks, the attacker aims to learn information about a record :math:`x`, either membership (:math:`x \in D^{(r)}`) or the value of a sensitive attribute :math:`s` (:math:`v~\text{s.t.}~x|v \in D^{(r)}`).

Summary
-------

.. list-table::
	:widths: 10 10 10 70
	:header-rows: 1

	* - Class
	  - Threat Model
	  - Parameters
	  - Decision uses
	* - ``ClosestDistanceMIA``
	  - MIA
	  - ``distance``, ``criterion``
	  - Distance of closest record to target record :math:`x` in :math:`D^{(s)}`.
	* - ``ClosestDistanceAIA``
	  - AIA
	  - ``distance``, ``criterion``
	  - For each value, distance of closest record to target record :math:`x|b` in :math:`D^{(s)}`.
	* - ``LocalNeighbourhoodAttack``
	  - MIA/AIA
	  - ``distance``, ``radius``, ``criterion``
	  - Sphere of given radius around the target record, :math:`B[x,radius]`. For MIA, use the fraction of records that are in the sphere. For AIA, use the fraction of records in the sphere with a given value :math:`v`.
	* - ``ShadowModellingAttack``
	  - MIA/AIA
	  - ``SetClassifier``
	  - Train a set classifier :math:`\mathcal{F}_\theta` from pairs :math:`(D^{(r)}_i, D^{(s)}_i)` to predict membership or the sensitive attribute for the sensitive data.
	* - ``GroundhogAttack``
	  - MIA/AIA
	  - ``features``, ``classifier``
	  - Shadow modelling attack using a random forest classifier over simple features extracted from datasets, proposed by Stadler et al. [1]_.
	* - ``ProbabilityEstimationAttack``
	  - MIA
	  - ``estimator``, ``criterion``
	  - Density estimator fit on synthetic records :math:`p_\theta`, and the density estimated in the target record :math:`p_\theta(x)`.
	* - ``SyntheticPredictorAttack``
	  - AIA
	  - ``estimator``, ``criterion``
	  - Classifier fit on synthetic records to predict the sensitive attribute :math:`x_s` from other attributes :math:`x_{-s}`.

Trainable-threshold attacks
---------------------------

Many attacks rely on a fixed (non-trained) score function :math:`s: (D^{(s)}, x) \rightarrow \mathbb{R}`. The decision (``.attack``) thus takes the form of a threshold on this score, :math:`\mathcal{A}(D^{(s)}, x) = 1 \Leftrightarrow s(D^{(s)}, x) \geq \tau`, i.e. predicting the positive class if and only if the score is above some threshold :math:`\tau`. This threshold can be selected empirically from observing a large enough set of training datasets for which the attack target (e.g., membership of :math:`x`) is known.

The partially abstract class ``TrainableThresholdAttack`` extends ``Attack`` to represent these score-based attacks. It implements the ``train`` and ``attack`` methods of ``Attack``, based on an attack-specific ``.attack_score`` to be implemented. Many of the attacks described below extend this class.
The constructor of these attacks takes an argument ``criterion`` that describes how the threshold should be selected. This argument should be a tuple, where the first entry describes the general criterion, and other entries provide additional information if needed. There are four options:

- ``criterion = ("accuracy",)``: the threshold is selected to approximately maximise the accuracy of ``.attack``.
- ``criterion = ("tp", value)``: the threshold is selected such that the true positive rate of the method is approximately equal to ``value``.
- ``criterion = ("fp", value)``: (similarly, but for the false positive rate).
- ``criterion = ("threshold", value)``: the threshold is set to ``value``. In this case, no further training is required.

*Note*: this attack generally applies to the black-box setting, but if ``criterion[0] = "threshold"``, then the attack can be applied *without* access to the generator (the no-box setting).


Closest-distance Attacks
------------------------

Closest-distance Attacks are *targeted attacks* that rely on the local neighbourhood of the target record in the synthetic data to infer information.
A simple example is a direct lookup attack, where the attacker predicts that the target record is in the real data if and only if it is also found in the *synthetic* data :math:`x \in D^{(s)}`.
They exploit the fact that real records have a higher likelihood to be found in the synthetic data, especially if the generation method is poorly designed (e.g., it is overfitted).
These attacks require a meaningful notion of distance between records in a dataset, represented by a ``tapas.attacks.DistanceMetric`` object. While ``TAPAS`` only implements two simple distances (``HammingDistance`` and ``LpDistance``), this object is little more than a wrapper over ``__call__`` and custom distance metrics are straightforward to implement.


ClosestDistanceMIA
~~~~~~~~~~~~~~~~~~

This membership inference attack uses the minimal distance between the target record and records in the synthetic dataset as (minus the) score: :math:`s(D^{(s)}, x) = - \min_{y \in D^{(s)}} distance(x, y)`. The negation of the distance is used instead, as a positive score indicates higher likelihood of membership.
This attack can be seen as a generalisation of direct-lookup attacks.

Parameters:

- ``distance``: a ``DistanceMetric`` object describing the distance to use between records.
- ``criterion`` (see `above <Trainable-threshold attacks>`_).


ClosestDistanceAIA
~~~~~~~~~~~~~~~~~~

Similarly, this attribute inference attack uses a score proportional to the minimal distance between the target record and records in the synthetic dataset, for each possible value of the sensitive attribute: :math:`s_v' = - \min_{y\in D^{(s)}} distance(x|v, y)`. The actual score is normalised, so as to give the *relative* distance for one value: :math:`s_v = \frac{s_v'}{\sum_u s_u'}`. In the case where there are only two values, the score is :math:`s := s_1 = \frac{s_1'}{s_1' + s_0'}`, and using a threshold of :math:`\tau = 0.5` is equivalent to choosing the value with smallest minimal distance.

Note that the above is more complex than the more intuitive idea of finding the record that is closest to the known attributes of the target record :math:`y \in \arg\min_{y\in D^{(s)}} x_{-s}` and using its value :math:`y_s` as answer. The approaches are however equivalent for any distance function that is space-invariant for the sensitive attribute, :math:`d(x|v, x|u) = f(v,u)~\forall x`, and more accurate for distances that do not satisfy this condition.

Parameters:

- ``distance``: a ``DistanceMetric`` object describing the distance to use between records.
- ``criterion`` (see `above <Trainable-threshold attacks>`_).


LocalNeighbourhoodAttack
~~~~~~~~~~~~~~~~~~~~~~~~

This attack, which can be used for both membership and attribute inference, uses the local neighbourhood of the target record in :math:`D^{(s)}` for the attack. This local neighbourhood is defined as the ball around `x` for a specific radius `r`, :math:`B[x,r] = \left\{y \in D^{(s)}: distance(x,y) \leq r\right\}`.

For membership inference, the score is the fraction of all records of :math:`D^{(s)}` that are also in :math:`B[x,r]`. The intuition is that if :math:`x` is in the real data, it is likely that similar records will be generated.

For attribute inference, the score for value :math:`v` is the fraction of records in :math:`B[x,r]` that have that value. Similarly, the idea is that if :math:`x` has value :math:`v` in the real data, then records similar to :math:`x` in the synthetic dataset are more likely to have value :math:`v`.

Parameters:

- ``distance``: a ``DistanceMetric`` object describing the distance to use between records.
- ``radius``: non-negative float, the radius of the local neighbourhood.
- ``criterion`` (see `above <Trainable-threshold attacks>`_).



Shadow Modelling Attacks
------------------------

Shadow modelling is a common technique to build privacy attacks against privacy-enhancing technologies.
The idea is to generate a large number of training "real" datasets :math:`(D_1^{(r)}, \dots, D_N^{(r)})` according to the attacker's knowledge (usually as subsets from an auxiliary dataset), then generate synthetic datasets from each of these: :math:`(D_1^{(s)}, \dots, D_N^{(s)})`.
For a function :math:`\phi` that the attacker is trying to learn (e.g., :math:`phi(D) = I\{x \in D\})`), they train a machine learning model :math:`\mathcal{F}_\theta` to infer the value of :math:`\phi` over real datasets from the synthetic dataset: :math:`\mathcal{F}_\theta(D^{(s)}) = \phi(D^{(r)})`.

The key design decision of a shadow modelling attack (``tapas.attacks.ShadowModellingAttack``) is in the choice of the classifier :math:`\mathcal{F}_\theta`.
A challenge of applying shadow modelling to synthetic datasets is that the input of the classifier is *the whole synthetic dataset*, and is thus very high-dimensional.
The first attack using shadow modelling for synthetic data is by Stadler et al.[1]_, an attack which we refer to as the Groundhog attack (``tapas.attacks.GroundhogAttack``).


ShadowModellingAttack
~~~~~~~~~~~~~~~~~~~~~

This class implements the logic of shadow modelling (in ``.train`` and ``.attack``) for membership and attribute inference attacks.
It takes one parameter, ``classifier``, a ``SetClassifier`` object that represents a classifier over *sets*.

A ``SetClassifier`` has an interface similar to ``scikit-learn`` classifiers, with ``.fit``, ``.predict`` and ``.predict_proba`` methods, except the inputs are lists of ``tapas.datasets.Dataset`` objects.


FeatureBasedSetClassifier
+++++++++++++++++++++++++

The main ``SetClassifier`` implemented by ``TAPAS`` is ``FeatureBasedSetClassifier``, a classifier that groups together two independent components:

1. ``features``: A ``SetFeature`` object that extracts a vector of features (a ``numpy.array``) from a ``Dataset``. This object is a fixed function :math:`psi` and is not trainable.
2. ``classifier``: A classifier from ``scikit-learn``, :math:`C_\theta`. This classifier is then trained (choosing :math:`\theta`) to infer the sensitive function :math:`\phi(D)` from the features extracted from a dataset.

The corresponding classifier is obtained by combining these two elements as :math:`\mathcal{F}_\theta = C_\theta \circ \psi`.

``SetFeature`` objects primarily consist of a ``.extract`` method mapping datasets to a ``numpy.array`` of size (len(datasets), k) for some size k. Implementing a custom ``SetFeature`` only requires to create an object inhering from ``tapas.attacks.SetFeature`` and defining the ``.extract`` method.
``TAPAS`` implements several simple ``SetFeature``.

.. list-table::
	:widths: 20 20 60
	:header-rows: 1

	* - Class
	  - Parameters
	  - Description
	* - ``NaiveSetFeature``
	  - /
	  - Computes the median, mean and variance of each column, with categorical columns 1-hot encoded. This is :math:`F_\text{naive}` from [1]_.
	* - ``HistSetFeature``
	  - :math:`n_\text{bins} =10`, :math:`bounds = (0,1)`
	  - Computes histograms for each attribute. For continuous attributes, the histogram is computed with :math:`n_\text{bins}`, over the interval :math:`bounds`. This is :math:`F_\text{hist}` from [1]_.
	* - ``CorrSetFeature``
	  - /
	  - Computes the correlation coefficient between all attributes, with categorical columns 1-hot encoded. This is :math:`F_\text{corr}` from [1]_.


GroundhogAttack
~~~~~~~~~~~~~~~

This class implements the attack from Stadler et al.[1]_. In ``TAPAS``, this is a ``FeatureBasedSetClassifier`` with, by default:

1. ``feature = NaiveSetFeature() + HistSetFeature() + CorrSetFeature()``
2. ``classifier = sklearn.ensemble.RandomForestClassifier()``.

The behaviour of this attack can be modified with four optional parameters:

- ``use_naive``: boolean, whether to use the Naive feature set (default True).
- ``use_hist``: boolean, whether to use the Histogram feature set (default True).
- ``use_corr``: boolean, whether to use the Correlations feature set (default True).
- ``model``: a ``scikit-learn`` classifier to use instead of the random forest (default None).


Inference-on-Synthetic Attacks
------------------------------

Inference-on-Synthetic attacks consist of attacks that make inference on the target record :math:`x` from a machine learning model trained on the synthetic data.
These are simple attacks based on traditional ``scikit-learn`` models, and are all instances of ``TrainableThresholdAttack``.

ProbabilityEstimationAttack
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Probability-Estimation attacks are membership inference attacks that use a density estimator fit to the synthetic data, :math:`p_\theta`. The score used by the attack is the density estimated in the target point, :math:`p_\theta(x)`. The intuition is that the presence of :math:`x` in the real dataset will bias the distribution from which synthetic records are sampled in such a way that synthetic records are more likely to "look like" :math:`x`.
The density estimated on the synthetic data can be seen as an approximation of the density of the generating distribution.

Parameters:

- ``estimator``: a ``DensityEstimator`` object, or a kernel density estimator from ``scikit-learn``.
- ``criterion`` (see `above <Trainable-threshold attacks>`_).

``DensityEstimator`` objects implement a ``.fit`` method, training parameters :math:`\theta` from a dataset, and a ``.score`` method, returning a density :math:`y\in\mathbb{R}` for records :math:`y`.
The main ``DensityEstimator`` provided by ``TAPAS`` is the internal class ``sklearnDensityEstimator``, which wraps a ``scikit-learn`` density estimator, and is used by the constructor of ``ProbabilityEstimationAttack``.


SyntheticPredictorAttack
~~~~~~~~~~~~~~~~~~~~~~~~

Synthetic predictor attacks are attribute inference attacks that train a machine learning model to predict the value of :math:`x_s` from known attributes :math:`x_{-s}`, :math:`C_\theta`, on records in the synthetic data. This is a common privacy attack, where correlations between attributes are exploited to predict the sensitive attribute (see, e.g., Correct Attribution Probability CAP [2]_). However, whether such attacks present a privacy risk is controversial, as an attacker can make a guess with accuracy better than random *even if* the target user is not in the dataset. ``TAPAS`` circumvents this issue by randomising the sensitive attribute independently from other attributes.

Disclaimer

- ``estimator``: a ``scikit-learn`` classifier to infer the sensitive attribute from other attributes. Categorical attributes of the data are 1-hot encoded before learning, so any classifier on real-valued data can be applied.
- ``criterion`` (see `above <Trainable-threshold attacks>`_).


References
----------

.. TODO: add the Usenix reference when the paper comes out.
.. [1] Stadler, T., Oprisanu, B. and Troncoso, C., 2021. Synthetic data–anonymisation groundhog day. arXiv preprint arXiv:2011.07018.
.. [2] Elliot, M., 2015. Final report on the disclosure risk associated with the synthetic data produced by the sylls team. Report 2015, 2.