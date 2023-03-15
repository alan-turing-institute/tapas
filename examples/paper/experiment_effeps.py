"""Script that runs a membership inference attack on three different generators."""

import tapas
import tapas.datasets
import tapas.threat_models

import tqdm

from tapas.generators import Generator, ReprosynGenerator

# Load the generator from the companion library (reprosyn).
from reprosyn.methods import MST

## Some parameters (see the paper).

# The privacy parameters.
epsilon = 10
delta = 1e-6
# The number of "other" (non-target) records.
training_dataset_size = 499
# Size of the output datasets (assumed known).
synthetic_dataset_size = 500
# The number of training and testing datasets for the attacks.
# (These numbers make it computationally expensive, try smaller numbers first)
num_training = 1000
num_testing = 2500


## Threat Modelling.
# We here use an optimisation within the library, which allows to save threat
# models to disk. This is useful because the threat model memorises all synthetic
# datasets generated for the training and testing. Saving to disk thus allows
# to try new attacks (or generate more samples) without having to generate
# all the datasets again.

threat_model_name = f"objects/effeps-mst-{epsilon}-{training_dataset_size}"

try:
    threat_model = tapas.threat_models.ThreatModel.load(threat_model_name)

except Exception as err:
    # No memorised file found: we thus define the threat model.
    print("Creating threat model...")

    # The threat model indirectly includes the dataset (through the auxiliary knowledge).
    # We first load the 1% census data.
    data = tapas.datasets.TabularDataset.read(
        "../data/2011 Census Microdata Teaching File", label="Census"
    )

    # We then select a target (outlier) record and remove it from the data.
    target_record_indices = [538738]
    target_record = data.get_records(target_record_indices)
    data.drop_records(target_record_indices, in_place=True)

    # We now select 499 records to serve as the rest of the dataset.
    specific_data = data.sample(n_samples=training_dataset_size)

    # We instantiate the generator (implemented in ReproSyn).
    # The label is used by TAPAS to generate human-friendly text in plots.
    generator = ReprosynGenerator(MST, label="MST", epsilon=epsilon, delta=delta)

    # Create the threat model for targeted MIA.
    threat_model = tapas.threat_models.TargetedMIA(
        # Attacker knowledge on the data: the attacker knows the entire dataset
        # except the membership of the target. This means that the prior over
        # the dataset outside of the target is deterministic.
        attacker_knowledge_data=tapas.threat_models.ExactDataKnowledge(specific_data),
        # Attacker knowledge on the generator: (exact) black-box knowledge.
        # Note: we assume that the attacker knows the size of the synthetic datasets.
        attacker_knowledge_generator=tapas.threat_models.BlackBoxKnowledge(
            generator, num_synthetic_records=synthetic_dataset_size,
        ),
        # Pass the target to the MIA (this will be combined with the knowledge
        # on the data to produce training datasets).
        target_record=target_record,
        # This "optimisation" can be used to force the training datasets to be
        # pairs (d, d u {t}) -- which only makes sense when there is randomness
        # over d. We disable it here.
        generate_pairs=False,
        # Whether to replace a random record with the target record. If False, the
        # target record is appended to the training dataset.
        replace_target=False,
        # Have a cute lil' display to track iterations.
        iterator_tracker=tqdm.tqdm,
    )
    threat_model.save(threat_model_name)


## Generate some samples to be cached by tapas.
# This is not a necessary step, but allows a neat separation between generation
# and attack training that is normally obscured by caching.
# Note that this is the most compute-intensive step.

threat_model._generate_samples(num_samples=num_training, training=True)
threat_model.save()
threat_model._generate_samples(num_samples=num_testing, training=False)
threat_model.save()


## Running attacks.
# We here apply a panoply of attacks from the library.

from tapas.attacks import *

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KernelDensity


attacks = [
    # Local-Neighbourhood Attack (with default distance, which is Hamming).
    ClosestDistanceMIA(criterion="accuracy", label="Closest-Distance"),
    # Inference-on-synthetic attack, with a Gaussian density kernel.
    ProbabilityEstimationAttack(
        KernelDensity(), criterion="accuracy", label="KernelDensity"
    ),
    # Shadow-modelling attacks.
    # This is vanilla Groundhog from Stadler et al.
    GroundhogAttack(label="GroundHog"),
    # Same features, but different classifier.
    GroundhogAttack(model=LogisticRegression(), label="LogisticGroundhog"),
    # This is a custom attack, that uses a mixture of queries of different order.
    ShadowModellingAttack(
        FeatureBasedSetClassifier(
            features=RandomTargetedQueryFeature(
                threat_model.target_record, order=2, number=500
            )
            + RandomTargetedQueryFeature(
                threat_model.target_record, order=3, number=500
            )
            + RandomTargetedQueryFeature(
                threat_model.target_record, order=1, number=20
            ),
            classifier=RandomForestClassifier(),
        ),
        label=f"RandomQueries",
    ),
]

## Apply the attacks.
# Running an attack produces a Summary object.

summaries = []

for attack in attacks:
    print(f"Training and testing {attack.label}...")
    # This uses the training samples from the threat model to train the attack.
    attack.train(threat_model, num_samples=num_training)
    # Then, the attack is "brought to" the threat model for evaluation.
    summaries.append(threat_model.test(attack, num_samples=num_testing))


## Produce nice reports.
# TAPAS supports a wide range of reporting functions. We will use the ROC curve
# and effective epsilons report.

from tapas.report import ROCReport, EffectiveEpsilonReport

report_name = f"reports/effeps-mst-{epsilon}-{training_dataset_size}"

# Create the repository for the reports, if it does not exist.
import os
if not os.path.exists(report_name):
    os.mkdir(report_name)

reports = [
    # Plots the ROC curve(s) for different attacks.
    ROCReport(
        summaries,
        # Suffix to add to the figure title (human readable).
        suffix="MST($\\varepsilon = 10$)",
        # A ROC curve will be generated for each "zoom" level (zooming around (0,0)).
        zooms=[1, 0.1],
        # In the ROC curve, we add linear bounds that correspond to effective epsilon = 2.
        eff_epsilon=2
    ),
    # This uses a heuristic to select an attack (using a fraction of test samples),
    # then uses this attack to estimate effective epsilon with Clopper-Pearson bounds.
    EffectiveEpsilonReport(summaries),
]

# Publishing means creating the files for the report.
for report in reports:
    report.publish(report_name)