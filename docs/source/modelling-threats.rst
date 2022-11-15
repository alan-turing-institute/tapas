=================
Modelling Threats
=================

Privacy evaluation with ``TAPAS`` relies on adversarial testing.
To test whether a synthetic data generation (SDG) technique provides privacy protection, ``TAPAS`` applies a range of attacks against it in order to try and infer sensitive information from synthetic datasets.
This requires a *threat model*: a representation of the assumptions made on what an attacker could know, and what they could be trying to infer.

In ``TAPAS``, threat models are represented with objects of the (abstract) class ``ThreatModel``. In its simplest form, this class just implements a ``.test`` method which takes a ``tapas.Attack`` object as argument. This method tests whether the attack succeeds, and returns some metric of success -- both of which depend on the context.
Typically, this method involves generating synthetic datasets from the real data and evaluating whether the attack can infer sensitive information from them.

Most commonly, threat models also hold some information that the attacker has access to. This information can be used to generate synthetic datasets on which to train the attack. These are represented as ``TrainableThreatModel``, an abstract class which defines an additional ``generate_training_samples`` method to produce synthetic datasets according to the attacker's auxiliary information.

In ``TAPAS``, threat models are composed of three independent components:

1. *Knowledge on the private dataset*.
2. *Knowledge on the generation method*.
3. *Attack goal*.

In the remainder of this page we explain how to describe a threat model using these three components.
As a quick overview of this framework, here is a generic attack that applies to a given dataset:

1. Sample **training** datasets that reflect knowledge on the private dataset: :math:`\tilde{D}_1, \dots, \tilde{D}_k`.
2. Generate **synthetic training** datasets from these training datasets, using knowledge of the generation method: :math:`\tilde{D}_i^{(s)} = G(\tilde{D}_i)`
3. Train a model :math:`\phi` to predict some sensitive function :math:`f` of the training dataset from the sensitive dataset: :math:`\phi(\tilde{D}_i^{(s)}) \approx f(\tilde{D}_i)`:
4. Apply this model to the **target synthetic dataset** :math:`D^{(s)}` to infer the value of the sensitive function over the **private dataset**: :math:`Attack(D^{(s)}) = \phi(D^{(s)}) \approx f(D)`

While this framework covers a variety of situations, you may find that it does not cover your specific use case. In that case, you will need to implement ``ThreatModel`` by hand, as well as attacks applied to it.


Knowledge on the training dataset
---------------------------------

The goal of publishing synthetic data is to prevent the disclosure of sensitive information from a private dataset `D`.
The attacker has *some* information on the private dataset. This information can be represented by a prior :math:`p_D` over datasets.

In `TAPAS`, the abstract class ``tapas.threat_models.AttackerKnowledgeOnData`` represents this knowledge. It defines a ``generate_datasets`` method which samples (private) datasets from the prior :math:`p_D`. Importantly, these are **training** datasets that reflect knowledge of the private datasets and are **not** synthetic datasets.

In practice, we disaggregate the data knowledge in two parts:

1. Information on the general dataset.
2. Information on sensitive elements, such as the presence of a target record.

The first is represented by some pre-defined ``AttackerKnowledgeOnData`` objects, which we explain below.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Class
     - Description
   * - ``AuxiliaryDataKnowledge``
     - The attacker has access to an auxiliary dataset, disjoint from the test dataset. The attacker is able to generate datasets from the same distribution as the real dataset. This is a common setup, e.g. found in Stadler et al.
   * - ``ExactDataKnowledge``
     - The attacker knows the exact private dataset. This is a "worst-case" attack model, typically used to audit privacy in the context of differential privacy.

The second part is performed by specific classes (which extend ``AttackerKnowledgeOnData``) that modify the datasets produced by other ``AttackerKnowledgeOnData`` objects to have a diversity of sensitive values. This is done on a case-by-case basis within the threat model classes (see below), and in general you don't need to worry about it.


Knowledge on the generation method
----------------------------------

The attacker is typically assumed to have some knowledge of the synthetic data generation method.
This is represented by a ``tapas.threat_models.AttackerKnowledgeOnGenerator``, that is mostly a wrapper over the generator.
``TAPAS`` implements a range of possible knowledge.

.. list-table::
    :widths: 20 80
    :header-rows: 1

    * - Class
      - Description
    * - ``NoBoxKnowledge``
      - The attacker does not know anything about the synthetic data generation procedure. They are thus not able to generate training synthetic datasets.
    * - ``BlackBoxKnowledge``
      - The attacker is able to run the exact synthetic data generation procedure (up to the choice of random seeds) on inputs of their choice, and observe outputs (synthetic datasets).
    * - ``WhiteBoxKnowledge``
      - Same as black-box, except that the attacker can additionally observe trained parameters which are used to produce the synthetic dataset. For instance, these could be the parameters of a generative network trained on the private data.
    * - ``UncertainBoxKnowledge``
      - The attacker knows that the generation procedure is a function :math:`G(\circ|\theta)` for some *unknown* parameter :math:`\theta \in \Theta`. The attacker also has a *prior*, a distribution over plausible values of this parameter :math:`p_\theta`.



Attack goal
-----------

Traditionally, privacy attacks attempt to *re-identify* a target record in the data.
In synthetic data, this is impossible: records in the synthetic data are *not* linked 1-1 with real records.
Attack models must thus detect different forms of leakages in the data.

``TAPAS`` implements two standard threat models, which have been applied to most types of data releases (e.g. aggregate statistics, machine learning models, synthetic data).

.. list-table::
    :widths: 15 20 65
    :header-rows: 1

    * - Class
      - Type
      - Description
    * - ``TargetedMIA``
      - **Membership Inference Attack**
      - The attacker attempts to infer whether the private dataset contains a *target* record :math:`x`.
    * - ``TargetedAIA``
      - **Attribute Inference Attack**
      - The attacker attempts to infer the value of the *sensitive* attribute :math:`a` of a *target* record :math:`x` in the private dataset. The attacker is assumed to know that the record is in the dataset, and the value of all other attributes :math:`x_{-a}`.

Both of these classes take the auxiliary knowledge on both the dataset and the mechanism as arguments to their constructors.
Under the hood, they wrap the ``AttackerKnowledgeOnData`` objects to randomly append/modify the target record to produce (labelled) training datasets.

Other attack models have been proposed in the literature, which will be implemented in future work:

- **Reconstruction attack**: the attacker attempts to produce a significant number of records from the private dataset.
- **Uniqueness attack**: the attacker attempts to identify whether a *target* record :math:`x` is unique in the dataset.
