Quick Start
===========

We here present an example on how to use ``TAPAS`` to apply attacks against a synthetic data generator and produce human-readable reports. The code on this page is taken (with slight changes) from the `groundhog_census.py <https://github.com/alan-turing-institute/privacy-sdg-toolbox/blob/main/examples/groundhog_census.py>`_ example.


Python Interface
----------------

For this example, we apply the `Groundhog attack by Stadler et al. <https://www.usenix.org/system/files/sec22summer_stadler.pdf>`_ to the Census 1% Teaching file of the England and Wales census and the simplest Raw generator.

The goal of ``TAPAS`` is to evaluate the possibility of an attack, rather than train and deploy attacks against real datasets. This informs the design decisions made, especially as relates to the auxiliary knowledge.

This example is meant as a general introduction to ``TAPAS``, and explains some important design choices to make when using the toolbox.

First we import the required dependences to run our attack (in this case, the various ``TAPAS`` modules).

.. code:: python

   import tapas.datasets
   import tapas.generators
   import tapas.threat_models
   import tapas.attacks
   import tapas.report

As a dataset, we are going to use the 1% teaching file from the 2011 England and Wales census, which is available to download at `this webpage <https://www.ons.gov.uk/census/2011census/2011censusdata/censusmicrodata/microdatateachingfile>`_.

TAPAS requires a json file that describes the columns of your (tabular) dataset. This allows TAPAS to understand the type of each of the columns, and which of the columns are discrete and which are continuous. For example for the residence type:

.. code:: json

   {
       "name": "Residence Type",
       "type": "finite",
       "representation": [
           "H",
           "C"
       ]
   },

For this quick start guide we have constructed a ``json`` file for this dataset, but for a full guide on creating a dataset for your use case please see the ``Data format`` page.

We place the dataset and description file in a ``data`` folder, with the following format:

.. code:: bash

   ├── data 
   │   ├── 2011 Census Microdata Teaching File.csv
   │   ├── 2011 Census Microdata Teaching File.json

TAPAS represents datasets as `tapas.datasets.TabularDataset` objects. These objects provide a large range of useful operations, and use ``pandas.DataFrame`` representations internally. We can load the 1\% census data in ``TAPAS`` using the following command (assuming the file and description are in the right folder):

.. code:: python


   data = tapas.datasets.TabularDataset.read("data/2011 Census Microdata Teaching File")

In this example, we will apply an attack against the ``Raw`` generator, which (trivially) outputs its training dataset. This is, of course, an insecure mechanism.

.. code:: python

   generator = tapas.generators.Raw()

We now define the threat model. The threat models is the core of ``TAPAS``: it encompasses all assumptions made on the attacker. A threat model has three components: (1) what the attacker knows about the *dataset*, (2) what the attacker knows about the *generator*, and (3) what they are trying to infer.

First, we define the attacker's knowledge about the dataset. Here, we assume that the attacker has access to an auxiliary dataset from the same distribution, from which they can sample training datasets.

.. code:: python

   data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(
         data,
         auxiliary_split=0.5,
         num_training_records=5000
   )

In this example, the attacker has access to 50% of the total dataset as auxiliary information. This information will be used to generate training datasets. The attacker knows that the real dataset contains 5000 samples. This thus reflects the attacker’s knowledge about the real data.

We then define what the attacker knows about the synthetic data generator. This would typically be black-box knowledge: the attacker is able to run the (exact) SDG model on any dataset that they choose but can only observe (input, output) pairs and not internal parameters. The attacker also specifies the size of the output (synthetic) dataset.

.. code:: python

   sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(
       generator,
       num_synthetic_records=5000,
   )


Finally, having defined the attacker’s knowledge, we specify their goal -- what they are trying to infer. We will here focus on a membership inference attack on a target record. We here (arbitrarily) select the first record in the dataset.

.. code:: python

   threat_model = tapas.threat_models.TargetedMIA(
      attacker_knowledge_data=data_knowledge,
      target_record=data.get_records([0]),
      attacker_knowledge_generator=sdg_knowledge,
      generate_pairs=True,
      replace_target=True
   )
                       

The first three parameters specify the components of the attacker knowledge that we defined above, and the target record. The last two parameters inform how the attacker will be trained (e.g. do we generate pairs :math:`(D, D U {target})` to train the attack). For full details please see the API documentation.

Defining and training an attack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The threat model represents what an attacker can do. We can now define an attacker whose abilities are allowed by the threat model.
In this example, we instantiate the GroundHog attack with standard parameters (from Stadler et al., 2022), which is provided as is within ``tapas``.

.. code:: python

   attacker = tapas.attacks.GroundhogAttack()

The groundhog attack is actually a particular instantiation of a larger class of attacks. It could equivalently be written using the following ``TAPAS`` classes (refer to the documentation for more details):

.. code:: python

   from sklearn.ensemble import RandomForestClassifier

   attacker = tapas.attacks.ShadowModellingAttack(
      tapas.attacks.FeatureBasedSetClassifier(
         tapas.attacks.NaiveSetFeature() + tapas.attacks.HistSetFeature() + tapas.attacks.CorrSetFeature(),
         RandomForestClassifier(n_estimators = 100)
      ),
      label = "Groundhog"
   )

In ``TAPAS``, attacks have to be trained for a specific threat model. This training involves (1) setting parameters linked to the dataset (such as the name of attributes and number of categories), and (2) the training of internal model parameters for the attack.
In the training phase, the attacker sees a large number of "real" datasets generated according to the attacker's knowledge and synthetic datasets generated from these real datasets.
These pairs can be used to, e.g., train a classifier to infer something about the real dataset from the synthetic dataset.
In this case, the classifier will be trained to infer whether a specific record is in the real dataset.
This is done using the ``.train`` method, which also requires to specify of number of training samples (pairs of real and synthetic datasets):

.. code:: python

   attacker.train(threat_model, num_samples=1000)

The ``TargetedMIA`` threat model is a ``TrainableThreatModel`` it defines a method to generate training samples (synthetic_dataset, target_in_real_dataset). This is why the threat model is passed to train the attacker. threat_model. The ``num_samples`` parameter is the number of training pairs generated by the threat model to train the attacker.

Evaluate the attack on the threat model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *evaluation* of the attack is done within the threat model object. This is because, conceptually, evaluation requires access to information that the attacker does not know (e.g., the remaining 50% of the dataset).
Evaluation functions similarly to training in that a large number of "real" and synthetic datasets are generated, the attack is applied to each synthetic datasets, and the success rate of the attack is measured.
This can be done with the ``threat_model.test`` function:

.. code:: python

   attack_summary = threat_model.test(attacker, num_samples = 100)


This outputs a ``tapas.report.MIAttackSummary`` object, that contains the labels of training "real" datasets and the label (and score) predicted by the attack. You can either use this summary directly (it contains a range of aggregate statistics):

.. code:: python

   metrics = attack_summary.get_metrics()
   print("Results:\n", metrics.head())


The resulting table then looks something like this (every run will be different since there is randomness in the threat model and attack training):

=================== ==========
Column              Value
=================== ==========
dataset             Census
target_id           0
generator           Raw
attack              GroundHog
accuracy            0.53
true_positive_rate  0.54
false_positive_rate 0.48
mia_advantage       0.06
privacy_gain        0.94
auc                 0.53
effective_epsilon   0.44
=================== ==========

Note that, in this case, the attack does not work very well. This is mostly due to the choice of features -- hence why we should apply a large range of attacks! See an example of this in `multiple_attacks.py <https://github.com/alan-turing-institute/privacy-sdg-toolbox/blob/main/examples/multiple_attacks.py>`_.

Alternatively, you can pass the summary on to a ``tapas.report.Report`` object. Report objects wrap the output of a summary, and can be published as files to interpret the success rate of the attack. The typical example is ``BinaryAIAttackReport``, which produces several plots to compare different generators, attacks, targets and metrics. This will save the output of the report to several files in the ``groundhog-census/`` folder.

.. code:: python

   report = tapas.report.BinaryAIAttackReport(
      [attack_summary], metrics=["accuracy", "privacy_gain", "auc"]
   ) 
   report.publish('groundhog-census')
