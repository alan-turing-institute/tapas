=================
Modelling Threats
=================

Privacy evaluation with `PrivE` relies on adversarial testing.
To test whether a synthetic data generation (SDG) technique provides privacy protection, `PrivE` applies a range of attacks against it in order to try and infer sensitive information from synthetic datasets.
This requires a *threat model*: a representation of the assumptions made on what an attacker could know, and what they could be trying to infer.

In `PrivE`, threat models are represented with objects of the (abstract) class `ThreatModel`. In its simplest form, this class just implements a `.test` method which takes a `prive.Attack` object as argument. This method tests whether the attack succeeds, and returns some metric of success -- both of which depend on the context.
`PrivE` implements a wide range of threat models and metrics, which we describe here.

`PrivE`'s threat models are composed of three components:

1. *Knowledge on the training dataset*.
2. *Knowledge on the generation method*.
3. *Attack goal*.


Knowledge on the training dataset
---------------------------------

`prive.threat_models.AttackerKnowledgeOnData`

Knowledge on the generation method
----------------------------------

`prive.threat_models.AttackerKnowledgeOnGenerator`

Attack goal
-----------

MIA
AIA