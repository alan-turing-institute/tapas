====================
Implementing Attacks
====================

Privacy attacks are at the core of ``TAPAS``.
In order to detect privacy leakage, ``TAPAS`` deploys a panoply of pre-implemented attacks, ranging in complexity and exploited vulnerability.
There are several reasons why one might want to implement attacks:

1. Attacks currently implemented in the toolbox fail against the model being evaluated.
2. The threat model considered is new and/or not well supported (existing attacks cannot be deployed against it).
3. You have a cool research idea you'd like to try and evaluate against a range of models.

Attacks are usually implemented with a (range of) threat model(s) and dataset(s) in mind.
For instance, some attacks assume a tabular dataset and will not apply to a time-series dataset, or specifically designed for membership inference attacks.

We here first explain the philosophy behind ``TAPAS``'s design of threat models and attacks, then how to implement a new attack.


Philosophy
----------

An attack can be modelled as a randomised function of a synthetic dataset :math:`\mathcal{A}_\theta: D^{(s)} \mapsto a \in \mathcal{S}`, possibly with some parameters :math:`\theta`.
The output of an attack aims at approximating a *sensitive* function :math:`f` of the *real* dataset, :math:`a \approx f\left(D^{(r)}\right)`.
The target function :math:`f`, and the quality of the approximation :math:`\approx` depend on the specific threat model.

In ``TAPAS``, an ``Attack`` is thus defined by two elements:

1. *Evaluation*: what the attack does, the "function call" of :math:`\mathcal{A}_\theta(\circ)`. This is defined by the ``.attack`` and ``.attack_score`` functions, taking a list of synthetic datasets as argument, and outputting a decision per dataset. The ``.attack`` method outputs a "decision" on :math:`f\left(D^{(r)}\right)`, while ``.attack_score`` computes the confidence of the decision.
2. *Training*: how to choose the parameters :math:`\theta` of the attack. This behaviour is defined by the ``.train`` method, which takes as input a threat model. This threat model captures all the information available to an attacker.

The evaluation of the success rate attack is performed by the ``ThreatModel`` object itself, through its ``.test`` function.
The ``.test`` function takes an ``Attack`` as argument, and returns a ``AttackSummary`` object of some description, depending on the attack type.
This allows for situations where the information available to the attacker (to train and perform the attack) is incomplete and/or incorrect.

Obviously, not all Attacks can be applied to all situations. Attacks are usually designed under specific assumptions on the attacker's goal and knowledge.
In particular, there are three dimensions to consider when designing an attack:

1. ``Dataset`` type: what minimal operations must the ``Dataset`` object support? Is the attack designed for a specific dataset type (e.g., ``TabularRecord``)?
2. ``Attack`` goal: what function :math:`f` of the real dataset is the attack trying to infer?
3. Attacker knowledge: what information about the real dataset is required for the attack? What information about the generator is required?

All of these aspects are defined by the ``ThreatModel`` object. A good design practice is to check in the ``.train`` method whether the attack can be applied in this threat model.



Implementing a new ``Attack``
-----------------------------

As discussed above, an ``Attack`` object requires the implementation of three methods:

1. ``.train(threat_model, **kwargs)``: checks that the threat model is compatible, and (optionally) set internal parameters from the information available to the attacker. See `below <Training an Attack_>`_  for more details.
2. ``.attack(datasets)``: performs the attack on each *synthetic* dataset, and returns a *decision* for each of them. The nature of this decision depends on the specific threat model. We discuss a common family of threat models `below <LabelInferenceThreatModel_>`_, where the decision is a countable value.
3. ``.attack_score(datasets)``: similar to ``.attack``, except a (typically real-valued) *score* is returned for each dataset. This score typically reflects the confidence of the attacker for each possible decision. The output of ``attack_score`` should be compatible with ``attack``. This method is *optional*: if the attack you implement does not have a meaningful notion of score, returning a constant score for all datasets is allowed. However, some reporting methods will not work if a score is not provided.

Additionally, the following methods are useful to implement:

1. The constructor, ``.__init__``, which should contain all parameters that describe the attack. Note that it is not recommended to pass data/``DataDescription`` to this constructor.
2. ``.label()``, a *property* that defines a user-readable string to use as name for this attack (e.g., in figures).

``TAPAS`` provides several tools to implement attacks, which we detail here.



Training an Attack
~~~~~~~~~~~~~~~~~~

Training an attack "from the information available to the attacker" seems fairly abstract. However, most threat models can be represented by the class ``TrainableThreatModel``, which extends ``ThreatModel`` with a ``.generate_training_samples(num_samples)`` method.
This function generates ``num_samples`` sample synthetic datasets on which to train the attack. In all generality, this method can output pairs of real and synthetic datasets :math:`\left(D^{(r)}_i, D^{(s)}_i\right)`, but typically only the value of :math:`f` on the real data is returned, :math:`\left(f\left(D^{(r)}_i\right), D^{(s)}_i\right)`.
These synthetic dataset samples can then be used to train the attack, e.g. using classical supervised learning methods.

How is the attacker knowledge used to generate these samples?

- Intuitively, the attacker knowledge on the dataset can be seen as a prior :math:`\pi_D`, from which training "real" datasets :math:`\left(D^{(r)}_1, \dots, D^{(r)}_{\text{num}_\text{samples}}\right)` can be sampled.
- Then, the attacker knowledge on the generator can be used to simulate the generator and produce training synthetic datasets :math:`D^{(s)}_i = \hat{G}(D^{(r)}_i)`.

See the documentation page on `Modelling Threats <modelling-threats.html>`_ for more details.



Training against a ThreatModels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Attacks are trained against a specific threat model. The threat model holds information related to the specific task at hand, which can be used for training. We here describe what parameters ``TAPAS`` threat models contain.

LabelInferenceThreatModel
+++++++++++++++++++++++++

Many common attacks can be modelled as *label-inference* attacks, where the decision set is finite (a *label*), :math:`\mathcal{S} = \{1, \dots, k\}`. Notably, this includes membership inference attacks (MIA), where the label is binary :math:`\mathcal{S} = \{0,1\}`, and attribute inference attacks (AIA), where the label can take any value allowed for the sensitive attribute :math:`s`: :math:`\mathcal{S} = \mathcal{V}_s`.

In ``TAPAS``, such threat models inherit from the class ``LabelInferenceThreatModel``, which implements ``.test`` and ``.generate_training_samples`` for attacks whose goal is to predict the label. The latter method is implemented to return a pair ``(synthetic_datasets, corresponding_labels)``.
The common attack classes, ``TargetedMIA`` and ``TargetedAIA``, inherit from this class, and implement the logic of MIA/AIA with purpose-specific ``AttackerKnowledgeWithLabel`` objects.
If the attack you are implementing is agnostic to the *semantics* of the label and can treat the attack task as a classification problem, you may have a label-inference attack.

In addition to ``.generate_training_samples``, these threat models have the following (mostly technical) parameters:

- ``atk_know_data``: an ``AttackerKnowledgeWithLabel`` object, that implements ``.generate_datasets_with_label``. Unless your attack is tailored to specific classes of attacker knowledge, refrain from using this explicitly.
- ``atk_know_gen``: an ``AttackerKnowledgeOnGenerator`` object, that implements ``.generate``. Similarly, unless your attack requires specific knowledge of the generator, refrain from using this explicitly.

TargetedMIA
+++++++++++

Targeted Membership Inference Attacks aim at inferring whether a specific *target* record :math:`x` is in the real data. Such threat models are implemented in ``TAPAS`` as ``LabelInferenceThreatModel`` where the label is membership of the target records, :math:`l = I\left\{x \in D^{(r)}\right\}`.
In addition to the attributes inherited from the parent, these threat models also have  the following attributes:

- ``target_record``: a ``Dataset`` object with one entry, the record of the target user.


TargetedAIA
+++++++++++

Targeted Attribute Inference Attacks aim at inferring the value of a *sensitive* attribute :math:`s` of a specific *target* record :math:`x`. Similarly, such threat models are ``LabelInferenceThreatModel`` objects, where the label is the value :math:`x_s \in \mathcal{V}_s`.
A key difference is that these threat models require the notion of attribute to be well-defined, and thus mostly apply to tabular datasets.
In addition to the attributes inherited from the parent, these threat models also have the following attributes:

- ``target_record``: a ``Dataset`` object with one entry, the record of the target user. The value of the sensitive attribute of this object is uninformative and should be ignored.
- ``sensitive_attribute``: the name (``str``) of the sensitive attribute.
- ``attribute_values``: a list of possible values for the sensitive attribute.



Trainable-Threshold Attacks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many binary label-inference attacks can be defined solely by a non-trainable score :math:`s: \mathcal{D} \mapsto \mathbb{R}`. The decision made by ``.attack`` is based on a threshold :math:`\tau`,  :math:`\mathcal{A}_\theta(D^{(s)}) = 1 \Leftrightarrow s(D^{(s)}) \geq \tau`.
Training the attack thus only involves *selecting a threshold* that leads to good results, according to some criterion.
``TAPAS`` provides a ``TrainableThresholdAttack`` class for these attacks, that only requires the attack designer to implement ``.attack_score``.
The constructor of these attacks has an additional parameter, a tuple ``criterion``, which defines how the threshold is selected.
There are several options, detailed in the documentation page on `Library of Attacks <library-of-attacks.html>`_.


ShadowModellingAttack
~~~~~~~~~~~~~~~~~~~~~

Shadow-modelling attacks are label-inference attacks where the attacker trains a classifier :math:`C_\theta` over synthetic datasets to predict the label of the real dataset. 
``TAPAS`` implements shadow-modelling attacks with the ``ShadowModellingAttack`` class. This class takes as argument a ``tapas.attacks.SetClassifier`` object.
If you wish to implement a shadow-modelling attack, the easiest way if to implement a custom ``SetClassifier`` object.
For more details on shadow-modelling attacks, see the documentation page on `Library of Attacks <library-of-attacks.html>`_.