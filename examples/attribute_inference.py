"""
This example is similar to multiple_attacks.py, except that it applies
attribute inference attacks (AIA) instead. AIAs are generally used in cases
where membership is not a meaningful threat.

The TAPAS interface for AIAs is similar to MIAs, except one must also specify
a sensitive attribute to guess, and a set of possible values. Many attacks
in TAPAS apply seemlessly to either AI and MI, so little change is required.

"""

import tapas.datasets
import tapas.generators
import tapas.threat_models
import tapas.attacks
import tapas.report

from sklearn.linear_model import LogisticRegression

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

threat_model = tapas.threat_models.TargetedAIA(
    attacker_knowledge_data=data_knowledge,
    # Specific to AIA: the sensitive attribute and its possible values.
    sensitive_attribute="Sex",
    attribute_values=["1", "2"],
    target_record=data.get_records([0]),
    attacker_knowledge_generator=sdg_knowledge,
)

# We here create a range of attacks to test.
attacks = [
    tapas.attacks.GroundhogAttack(),
    tapas.attacks.ClosestDistanceAIA(criterion="accuracy"),
    tapas.attacks.ClosestDistanceAIA(
        distance=tapas.attacks.LpDistance(2), criterion="accuracy"
    ),
    tapas.attacks.LocalNeighbourhoodAttack(radius=1),
    tapas.attacks.LocalNeighbourhoodAttack(radius=2),
    tapas.attacks.SyntheticPredictorAttack(LogisticRegression(), criterion="accuracy"),
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
report = tapas.report.MIAttackReport(summaries)  # TODO: fix.
report.publish("multiple_aia")
