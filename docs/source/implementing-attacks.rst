====================
Implementing Attacks
====================

Privacy attacks are at the core of ``PrivE``.
In order to detect privacy leakage, ``PrivE`` deploys a panoply of pre-implemented attacks, ranging in complexity and exploited vulnerability.
There are several reasons why one might want to implement attacks:

1. Attacks currently implemented in the toolbox fail against the model being evaluated.
2. The threat model considered is new and/or not well supported (existing attacks cannot be deployed against it).
3. You have a cool research idea you'd like to try and evaluate against a range of models.

Attacks are usually implemented with a (range of) threat model(s) and dataset(s) in mind.
For instance, some attacks assume a tabular dataset and will not apply to a time-series dataset, or specifically designed for membership inference attacks.

We here first explain the philosophy behind ``PrivE``'s design of threat models and attacks, then how to implement a new attack.

Philosophy
----------

The ``Attack`` class 
``train``

``attack/attack_score``

The ``ThreatModel`` class has ``test``

``LabelInferenceThreatModel``


Under the hood
--------------

Imnplementing an attack involves implementing three 


The attack can also have a constructor ``__init__`` that defines parameters.

This *can* take a dataset description as argument, although it is better not to.

``generate_training_samples``



Set Classifiers
---------------

For attacks based on *shadow modelling*, the base class of

``GroundhogAttack``