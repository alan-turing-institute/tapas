"""
This example is similar to multiple_attacks.py, except that it applies
attribute inference attacks (AIA) instead. AIAs are generally used in cases
where membership is not a meaningful threat.

The PrivE interface for AIAs is similar to MIAs, except one must also specify
a sensitive attribute to guess, and a set of possible values. Many attacks
in PrivE apply seemlessly to either AI and MI, so little change is required.

"""

import prive.datasets
import prive.generators
import prive.threat_models
import prive.attacks
import prive.report

from sklearn.linear_model import LogisticRegression

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

threat_model = prive.threat_models.TargetedAIA(
    attacker_knowledge_data=data_knowledge,
    # Specific to AIA: the sensitive attribute and its possible values.
    sensitive_attribute="Sex",
    attribute_values=["1", "2"],
    target_record=data.get_records([0]),
    attacker_knowledge_generator=sdg_knowledge,
)

# We here create a range of attacks to test.
attacks = [
    prive.attacks.GroundhogAttack(),
    prive.attacks.ClosestDistanceAIA(criterion="accuracy"),
    prive.attacks.ClosestDistanceAIA(
        distance=prive.attacks.LpDistance(2), criterion="accuracy"
    ),
    prive.attacks.LocalNeighbourhoodAttack(radius=1),
    prive.attacks.LocalNeighbourhoodAttack(radius=2),
    prive.attacks.SyntheticPredictorAttack(LogisticRegression(), criterion="accuracy"),
]

# Train, evaluate, and summarise all attacks.
summaries = []
for attack in attacks:
    print(f"Evaluating attack {attack.label}...")
    attack.train(threat_model, num_samples=100)
    summaries.append(threat_model.test(attack, num_samples=100))
    print(summaries[-1].get_metrics())

# Finally, group together the summaries as a report.
print("Publishing a report.")
report = prive.report.MIAttackReport(summaries)  # TODO: fix.
report.create_report("multiple_aia")
