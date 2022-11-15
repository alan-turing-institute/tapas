"""
This example applies several attacks to tthe setup of groundhog_census.py, and
shows how to write code without duplicating unnecessary elements, and how to
handle reports with multiple attacks.

"""

import tapas.datasets
import tapas.generators
import tapas.threat_models
import tapas.attacks
import tapas.report

from tapas.attacks import LpDistance

import pandas

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KernelDensity

# Load the data.
data = tapas.datasets.TabularDataset.read(
    "data/2011 Census Microdata Teaching File", label="Census"
)

# Create a dummy generator.
generator = tapas.generators.Raw()

# Select the auxiliary data + black-box attack model.
data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(
    data, auxiliary_split=0.5, num_training_records=1000,
)

sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(
    generator, num_synthetic_records=1000,
)

threat_model = tapas.threat_models.TargetedMIA(
    attacker_knowledge_data=data_knowledge,
    target_record=data.get_records([0]),
    attacker_knowledge_generator=sdg_knowledge,
    generate_pairs=True,
    replace_target=True,
)


# We here create a range of attacks to test.
attacks = [
    tapas.attacks.GroundhogAttack(
        use_hist=False, use_corr=False, label="NaiveGroundhog"
    ),
    tapas.attacks.GroundhogAttack(
        use_naive=False, use_corr=False, label="HistGroundhog"
    ),
    tapas.attacks.GroundhogAttack(
        use_naive=False, use_hist=False, label="CorrGroundhog"
    ),
    tapas.attacks.GroundhogAttack(
        model=LogisticRegression(), label="LogisticGroundhog"
    ),
    tapas.attacks.ClosestDistanceMIA(
        criterion="accuracy", label="ClosestDistance-Hamming"
    ),
    tapas.attacks.ClosestDistanceMIA(
        criterion=("threshold", 0), label="Direct Lookup"
    ),
    tapas.attacks.ClosestDistanceMIA(
        distance=LpDistance(2), criterion="accuracy", label="ClosestDistance-L2"
    ),
    tapas.attacks.ProbabilityEstimationAttack(
        KernelDensity(), criterion="accuracy", label="KernelEstimator"
    ),
]

# Train, evaluate, and summarise all attacks.
summaries = []
for attack in attacks:
    print(f"Evaluating attack {attack.label}...")
    attack.train(threat_model, num_samples=100)
    summaries.append(threat_model.test(attack, num_samples=100))

# Finally, group together the summaries as a report.
print("Publishing a report.")
report = tapas.report.MIAttackReport(summaries)
report.publish("multiple_mia")

# Also publish the ROC curve.
report = tapas.report.ROCReport(summaries)
report.publish("multiple_mia")
