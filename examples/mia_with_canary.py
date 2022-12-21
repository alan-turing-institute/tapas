from tapas.datasets.canary import create_canary

import tapas.datasets
import tapas.generators
import tapas.threat_models
import tapas.attacks
import tapas.report

# Load the data.
real_data = tapas.datasets.TabularDataset.read(
    "data/2011 Census Microdata Teaching File", label="Census"
)

# Create a *canary* record: this is a record with unique values for each
# attribute. Creating this canary modifies the data description, and a new
# dataset object (with the same content) is thus generated.
data, canary = create_canary(real_data)

canary.set_id(-1)

# For comparison, we are also going to select a random record (0) as target.
target = data.get_records([0])
data.drop_records(record_ids=[0], in_place=True)

# Since the target is a dataset, we can add the canary record to it to form
# a dataset of two targets.
targets = target.add_records(canary)

generator = tapas.generators.Raw()

# We use the same setup as groundhog_census.py.
sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(
    generator, num_synthetic_records=5000,
)

data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(
    data, auxiliary_split=0.5, num_training_records=5000,
)

threat_model = tapas.threat_models.TargetedMIA(
    attacker_knowledge_data=data_knowledge,
    target_record=targets,  # except with two targets!
    attacker_knowledge_generator=sdg_knowledge,
    replace_target=True,
)

# By iterating over a threat model, we cycle through the target users.
# You can manually modify which user the threat model is trying to attack, but
# iterating automatically does it for you.

summaries = []
for threat_model_for_a_target in threat_model:
    attacker = tapas.attacks.GroundhogAttack()
    attacker.train(
        threat_model_for_a_target, num_samples=100,
    )
    summaries.append(threat_model_for_a_target.test(attacker))

# Finally, group together the summaries as a report.
reports = [tapas.report.MIAttackReport(summaries), tapas.report.ROCReport(summaries)]

for report in reports:
    report.publish("canary_mia")
