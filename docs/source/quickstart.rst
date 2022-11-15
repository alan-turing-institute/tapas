Quick start
===========

TO DO

Python Interface
----------------

For this example, we apply the `Groundhog attack <https://www.usenix.org/system/files/sec22summer_stadler.pdf>`_ to the Census 1%
Teaching file of the England and Wales census and the simplest Raw
generator.

The goal of TAPAS is to evaluate the possibility of an attack, rather
than train and deploy attacks against real datasets. This informs the
design decisions made, especially as relates to the auxiliary knowledge.

This example is meant as a general introduction to TAPAS, and explains
some important design choices to make when using the toolbox.

For those who would just like to run the attack we also provide a
directly executable version of this code in the github repo under
examples.

First we import the required dependences to run our attack.

.. code:: python

   import tapas.datasets
   import tapas.generators
   import tapas.threat_models
   import tapas.attacks
   import tapas.report

   # Some fancy displays when training/testing.
   import tqdm

Next we need to download the 1% Census Microdata file, which is
available at `this
webpage. <https://www.ons.gov.uk/census/2011census/2011censusdata/censusmicrodata/microdatateachingfile>`__

We additionally require a json description file which describes the
columns of your dataset. This allows our code to understand the type of
each of the columns, and which of the columns are discrete and which are
continuous. For example for the residence type:

.. code:: json

   {
       "name": "Residence Type",
       "type": "finite",
       "representation": [
           "H",
           "C"
       ]
   },

For this quick start guide we have constructed a ``json`` file for this
dataset, but for a full guide on creating a dataset for your use case
please see the ``Data format`` page.

We place the data in the following format:

.. code:: bash

   ├── data 
   │   ├── 2011 Census Microdata Teaching File.csv
   │   ├── 2011 Census Microdata Teaching File.json

We can load the data using the following command:

.. code:: python


   data = tapas.datasets.TabularDataset.read("data/2011 Census Microdata Teaching File")

We attack the (trivial) Raw generator, which outputs its training
dataset.

.. code:: python


   generator = tapas.generators.Raw()

We now define the threat model: what is assumed that the attacker knows.
We first define what the attacker knows about the dataset. Here, we
assume they have access to an auxiliary dataset from the same
distribution.

.. code:: python


   data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(data,
                       sample_real_frac=0.5, num_training_records=5000,)

In this example, the attacker has access to 50% of the data as auxiliary
information. This information will be used to generate training
datasets. The attacker knows that the real dataset contains 5000
samples. This thus reflects the attacker’s knowledge about the real
data.

We then define what the attacker knows about the synthetic data
generator. This would typically be black-box knowledge, where they are
able to run the (exact) SDG model on any dataset that they choose, but
can only observe (input, output) pairs and not internal parameters.

.. code:: python


   sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(generator, num_synthetic_records=5000, )

The attacker also specifies the size of the output dataset. In practice,
use the size of the published synthetic dataset.

Now that we have defined the attacker’s knowledge, we define their goal.
We will here focus on a membership inference attack on a random record.

We here select the first record, arbitrarily:

.. code:: python


   threat_model = tapas.threat_models.TargetedMIA(attacker_knowledge_data=data_knowledge,
                       target_record=data.get_records([0]),
                       attacker_knowledge_generator=sdg_knowledge,
                       generate_pairs=True,
                       replace_target=True,
                       iterator_tracker=tqdm.tqdm)
                       

The remaining options inform how the attacker will be trained (e.g. do
we generate pairs (D, D U {target}) to train the attack). For full
details please see the API documentation.

Next step: initialise an attacker in this example, we just apply the
GroundHog attack with standard parameters (from Stadler et al., 2022).
The GroundhogAttack attacker is mostly a wrapper over a set classifier.

.. code:: python


   from sklearn.ensemble import RandomForestClassifier

   rf = RandomForestClassifier(n_estimators=100)
   features = tapas.attacks.NaiveSetFeature() + tapas.attacks.HistSetFeature() + tapas.attacks.CorrSetFeature()
   fsClassfr = tapas.attacks.FeatureBasedSetClassifier(features, rf) 

   attacker = tapas.attacks.GroundhogAttack(fsClassfr)

We here use, as in Stadler et al., a feature-based set classifier, which
computes a vector of (fixed) features of the set to classify. We use the
F_naive, F_hist and F_corr features (from the paper). We use a random
forest with 100 trees and default parameters and initialise a
vector-based classifier using these features.

Having defined all the objects that we need, we can train the attack.

.. code:: python


   attacker.train(num_samples=1000)

The ``TargetedMIA`` threat model is a ``TrainableThreatModel`` it
defines a method to generate training samples (synthetic_dataset,
target_in_real_dataset). This is why the threat model is passed to train
the attacker. threat_model. The ``num_samples`` parameter is the number
of training pairs generated by the threat model to train the attacker.

Evaluate with the test model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can now test the model using the ``threat_model.test`` function, we
do so like this:

.. code:: python


   attack_labels, truth_labels = threat_model.test(attacker, num_samples=1000)

Finally, generate a report to evaluate the results. The summary requires
these attack_labels, truth_labels, And some metadata for nicer displays.

.. code:: python


   attack_summary = tapas.report.MIAttackSummary(generator_info="raw", attack_info=attacker.__name__, dataset_info="Census", target_id="0" )
   metrics = attack_summary.get_metrics() 
   print(metrics)

The resultant table then looks like this:

=================== ==================================
Column              Value
=================== ==================================
dataset             Census
target_id           0
generator           raw
attack              FeatureBasedSetClassifierGroundhog
accuracy            0.5285
true_positive_rate  0.531284
false_positive_rate 0.473829
mia_advantage       0.057455
privacy_gain        0.942545
=================== ==================================

Where we see that privacy gain is actually larger than we would expect, given
that the generator does not protect privacy at all, to alleviate this we would need to 
modify the statistics used in the attack. 
