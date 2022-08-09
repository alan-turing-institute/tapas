==================
Library of Attacks
==================

Adversarial approaches for privacy require auditors to run a large range of diverse attacks, in order to test for as many potential vulnerabilities as possible.
For this reason, ``PrivE`` implements a large range of attacks.
While some of these attacks are technically involved, many are straightforward and are intended mostly as safety checks.
We here present the different attacks implemented in ``PrivE``, grouped by theme. For each attack, we specify the additional parameters it requires, and the attack models (see `Modelling Threats <modelling-threats.rst>`) it applies to.
``PrivE`` attacks inherit from the ``prive.attacks.Attack`` abstract class (see `Implementing Attacks <implementing-attacks.rst>` for details).
Note that the constuctor of *all* the attacks described below allows for an optional ``label`` parameters, which we exclude from the descriptions for the sake of concision.

*Notations*: we denote by :math:`D^{(r)}` the real, private dataset, and :math:`D^{(s)}` the synthetic dataset obtained with the generation method :math:`\mathcal{G}`. For targeted attacks, the attacker aims to learn information about a record :math:`x`, either membership (:math:`x \in D^{(r)}`) or the value of a sensitive attribute :math:`s` (:math:`v~\text{s.t.}~x|v \in D^{(r)}`).


Trainable-threshold attacks
---------------------------

Many attacks rely on a fixed (non-trained) score function :math:`s: (D^{(s)}, x) \rightarrow \mathbb{R}`. The decision (``.attack``) thus takes the form of a threshold on this score, :math:`\mathcal{A}(D^{(s)}, x) = 1 \Leftrightarrow s(D^{(s)}, x) \geq \tau`, i.e. predicting the positive class if and only if the score is above some threshold :math:`\tau`. This threshold can be selected empirically from observing a large enough set of training datasets for which the attack target (e.g., membership of :math:`x`) is known.

The partially abstract class ``TrainableThresholdAttack`` extends ``Attack`` to represent these score-based attacks. It implements the ``train`` and ``attack`` methods of ``Attack``, based on an attack-specific ``.attack_score`` to be implemented. Many of the attacks described below extend this class.
The constructor of these attacks takes an argument ``criterion`` that describes how the threshold should be selected. This argument should be a tuple, where the first entry describes the general criterion, and other entries provide additional information if needed. There are four options:

- ``criterion = ("accuracy",)``: the threshold is selected to approximately maximise the accuracy of ``.attack``.
- ``criterion = ("tp", value)``: the threshold is selected such that the true positive rate of the method is approximately equal to ``value``.
- ``criterion = ("fp", value)``: (similarly, but for the false positive rate).
- ``criterion = ("threshold", value)``: the threshold is set to ``value``. In this case, no further training is required.


Closest-distance Attacks
------------------------

Closest-distance Attacks are *targeted attacks* that rely on the local neighbourhood of the target record in the synthetic data to infer information.
A simple example is a direct lookup attack, where the attacker predicts that the target record is in the real data if and only if it is also found in the *synthetic* data :math:`x \in D^{(s)}`.
They exploit the fact that real records have a higher likelihood to be found in the synthetic data, especially if the generation method is poorly designed (e.g., it is overfitted).
These attacks require a meaningful notion of distance between records in a dataset, represented by a ``prive.attacks.DistanceMetric`` object. While ``PrivE`` only implements two simple distances (``HammingDistance`` and ``LpDistance``), this object is little more than a wrapper over ``__call__`` and custom distance metrics are straightforward to implement.


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
- ``radius``: nonnegative float, the radius of the local neighbourhood.
- ``criterion`` (see `above <Trainable-threshold attacks>`_).



Shadow Modelling Attacks
------------------------

Shadow modelling is a technique th
Intuitively, 
This technique has been used for synthetic data in the paper by Stadler et al.
Groundhog [1]_

ShadowModellingAttack
~~~~~~~~~~~~~~~~~~~~~

Requires

GroundhogAttack
~~~~~~~~~~~~~~~

Inference-on-Synthetic Attacks
------------------------------

Link to CAP [2]_


References
----------

.. TODO: add the Usenix reference when it comes out.
.. [1] Stadler, T., Oprisanu, B. and Troncoso, C., 2021. Synthetic data–anonymisation groundhog day. arXiv preprint arXiv:2011.07018.
.. [2] Elliot, M., 2015. Final report on the disclosure risk associated with the synthetic data produced by the sylls team. Report 2015, 2.