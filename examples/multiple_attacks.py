"""
This example applies several attacks to tthe setup of groundhog_census.py, and
shows how to write code without duplicating unnecessary elements, and how to
handle reports with multiple attacks.

"""

import prive.datasets
import prive.generators
import prive.threat_models
import prive.attacks
import prive.report

from prive.attacks import LpDistance

import pandas

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KernelDensity

# Load the data.
data = prive.datasets.TabularDataset.read(
    "data/2011 Census Microdata Teaching File", label="Census"
)

# Create a dummy generator.
generator = prive.generators.Raw()

# Select the auxiliary data + black-box attack model.
data_knowledge = prive.threat_models.AuxiliaryDataKnowledge(
    data, auxiliary_split=0.5, num_training_records=1000,
)

sdg_knowledge = prive.threat_models.BlackBoxKnowledge(
    generator, num_synthetic_records=1000,
)

threat_model = prive.threat_models.TargetedMIA(
    attacker_knowledge_data=data_knowledge,
    target_record=data.get_records([0]),
    attacker_knowledge_generator=sdg_knowledge,
    generate_pairs=True,
    replace_target=True,
)


# We here create a range of attacks to test.
attacks = [
    prive.attacks.GroundhogAttack(
        use_hist=False, use_corr=False, label="NaiveGroundhog"
    ),
    prive.attacks.GroundhogAttack(
        use_naive=False, use_corr=False, label="HistGroundhog"
    ),
    prive.attacks.GroundhogAttack(
        use_naive=False, use_hist=False, label="CorrGroundhog"
    ),
    prive.attacks.GroundhogAttack(
        model=LogisticRegression(), label="LogisticGroundhog"
    ),
    prive.attacks.ClosestDistanceMIA(
        criterion="accuracy", label="ClosestDistance-Hamming"
    ),
    prive.attacks.ClosestDistanceMIA(
        criterion=("threshold", 0), label="Direct Lookup"
    ),
    prive.attacks.ClosestDistanceMIA(
        distance=LpDistance(2), criterion="accuracy", label="ClosestDistance-L2"
    ),
    prive.attacks.ProbabilityEstimationAttack(
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
report = prive.report.MIAttackReport(summaries)
report.create_report("multiple_mia")

# Also publish the ROC curve.
report = prive.report.ROCReport(summaries)
report.publish("multiple_mia")
