"""Script that runs an attribute inference attack on three different generators."""

import tapas
import tapas.datasets
import tapas.threat_models

import tqdm

# Load the generators from the companion library (reprosyn).
from tapas.generators import Generator, ReprosynGenerator
from reprosyn.methods import MST, CTGAN, DS_PRIVBAYES


## Some parameters (see the paper).
epsilon = 10
delta = 1e-5
training_dataset_size = 5000

# TODO: set 100 for ctgan.
num_trainings = [100, 100, 100]
num_testings = [100, 100, 100]


## Load the 1% census data.
data = tapas.datasets.TabularDataset.read(
    "../data/2011 Census Microdata Teaching File", label="Census"
)

## From the outlier analysis, this seems to be an outlier.
# Target record: 538738
# LogLikelihood: -39.92051644253627

# Select a target record and remove it.
target_record_indices = [538738]
target_record = data.get_records(target_record_indices)
data.drop_records(target_record_indices, in_place=True)


## Define the generators.

# We can set the label of the generators if needed.
config = {"epsilon": epsilon, "delta": delta}

generators = [
    ReprosynGenerator(DS_PRIVBAYES, label="PrivBayes", epsilon=epsilon),
    ReprosynGenerator(MST, label="MST", **config),
    ReprosynGenerator(CTGAN, label="CTGAN"),
]

## Threat Modelling.
# We here use an optimisation within the library, which allows to save threat
# models to disk. This is useful because the threat model memorises all synthetic
# datasets generated for the training and testing. Saving to disk thus allows
# to try new attacks (or generate more samples) without having to generate
# all the datasets again.

threat_models = {}

for generator in generators:
    threat_model_name = f"objects/aia_{generator.label}"
    try:
        threat_model = tapas.threat_models.ThreatModel.load(threat_model_name)
    except Exception as err:
        print(f"Threat model not found for {generator.label}, creating one ({err}).")
        # Create the threat model for targeted AIA.
        threat_model = tapas.threat_models.TargetedAIA(
            # Attacker knowledge on the data: the attack has access to an auxiliary
            # dataset (here, 50% of the total dataset) disjoint from the real
            # dataset(s) (here, we sample testing datasets from the remaining
            # 50% of the data). Training datasets are samples from this.
            attacker_knowledge_data=tapas.threat_models.AuxiliaryDataKnowledge(
                data, auxiliary_split=0.5, num_training_records=training_dataset_size,
            ),
            # Attacker knowledge on the generator: (exact) black-box knowledge.
            # Note: we assume that the attacker knows the size of the synthetic datasets.
            attacker_knowledge_generator=tapas.threat_models.BlackBoxKnowledge(
                generator, num_synthetic_records=training_dataset_size,
            ),
            # Pass the target to the AIA (this will be combined with the knowledge
            # on the data to produce training datasets).
            target_record=target_record,
            # Attribute to infer: Sex, with values 1 and 2.
            sensitive_attribute="Sex",
            attribute_values=["1", "2"],
            # Have a cute lil' display to track iterations.
            iterator_tracker=tqdm.tqdm,
        )
        threat_model.save(threat_model_name)
    # Save the resulting threat model.
    threat_models[generator.label] = threat_model


## Generate some samples to be cached by tapas.
# This is not a necessary step, but allows a neat separation between generation
# and attack training that is normally obscured by caching.
# Note that this is the most compute-intensive step.

for generator, ntr, nts in zip(generators, num_trainings, num_testings):
    print("Running", generator.label)
    threat_model = threat_models[generator.label]
    threat_model._generate_samples(num_samples=ntr, training=True)
    threat_model.save()
    threat_model._generate_samples(num_samples=nts, training=False)
    threat_model.save()


## Running attacks.
# We here apply a panoply of attacks from the library.

from tapas.attacks import *

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KernelDensity


attacks = [
    # Local-Neighbourhood Attack (with default distance, which is Hamming).
    ClosestDistanceAIA(criterion="accuracy", label="Closest-Distance"),
    # Inference-on-synthetic attack, with a random forest classifier.
    SyntheticPredictorAttack(
        RandomForestClassifier(), criterion="accuracy", label="SyntheticPredictor"
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
# We also collect the summaries disaggregated by generator.
summaries_per_generator = {}

for generator, ntr, nts in zip(generators, num_trainings, num_testings):
    tm = threat_models[generator.label]
    l = []
    for attack in attacks:
        print("Training+Testing", attack.label, "on", generator.label)
        # This uses the training samples from the threat model to train the attack.
        attack.train(tm, num_samples=ntr)
        # Then, the attack is "brought to" the threat model for evaluation.
        s = tm.test(attack, num_samples=nts)
        l.append(s)
    summaries_per_generator[generator.label] = l
    summaries += l


## Produce nice reports.
# TAPAS supports a wide range of reporting functions. We will use the binary
# AIA Reporting (which produces a range of metrics and plots disaggregated
# by attack/metric/generator) and the ROC curves.

from tapas.report import BinaryAIAttackReport, ROCReport


report_name = "reports/aia"

reports = [
    BinaryAIAttackReport(summaries, metrics=["accuracy", "privacy_gain", "auc"])
] + [
    ROCReport(summaries_per_generator[generator.label], suffix=f"{generator.label}")
    for generator in generators
]

for report in reports:
    report.publish(report_name)