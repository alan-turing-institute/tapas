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

# Load the data.
data = prive.datasets.TabularDataset.read("data/2011 Census Microdata Teaching File")

# Create a dummy generator.
generator = prive.generators.Raw()

# Select the auxiliary data + black-box attack model.
data_knowledge = prive.threat_models.AuxiliaryDataKnowledge(
    data, sample_real_frac=0.5, num_training_records=1000,
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

# We define a simple wrapper for groundhog, since it's otherwise a bit wordy.
groundhog = lambda features, model: prive.attacks.GroundhogAttack(
    prive.attacks.FeatureBasedSetClassifier(features, model)
)
# We here create a range of attacks to test.
attacks = [
    groundhog(
        prive.attacks.NaiveSetFeature(), RandomForestClassifier(n_estimators=100)
    ),
    groundhog(prive.attacks.HistSetFeature(), RandomForestClassifier(n_estimators=100)),
    groundhog(prive.attacks.CorrSetFeature(), RandomForestClassifier(n_estimators=100)),
    groundhog(prive.attacks.CorrSetFeature(), LogisticRegression()),
    # prive.attacks.ClosestDistanceAttack(criterion="accuracy"),
    prive.attacks.ClosestDistanceAttack(distance=LpDistance(2), criterion="accuracy")
]

attack_names = [
    "NaiveGroundhog",
    "HistGroundhog",
    "CorrGroundhog",
    "LogisticGroundhog",
    "Closest-Distance",
]

# Train, evaluate, and summarise all attacks.
summaries = []
for attack, name in zip(attacks, attack_names):
    print(f"Evaluating attack {name}...")
    attack.train(threat_model, num_samples=100)
    attack_labels, truth_labels = threat_model.test(attack, num_samples=100)
    summaries.append(
        prive.report.MIAttackSummary(
            attack_labels,
            truth_labels,
            generator_info="raw",
            attack_info=name,
            dataset_info="Census",
            target_id="0",
        ).get_metrics()
    )

# Finally, group together the summaries as a report.
print("Publishing a report.")
report = prive.report.MIAttackReport(pandas.concat(summaries))
report.create_report("multiple_attacks_raw")
