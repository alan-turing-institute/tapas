from datetime import datetime
import os

import numpy as np

from tapas.attacks.set_classifiers import LRClassifier, RFClassifier, \
    SetReprClassifier, NaiveRep
from tapas.threat_models.mia import TargetedAuxiliaryDataMIA
from tapas.attacks import Groundhog, ClosestDistanceMIA
from tapas.datasets import TabularDataset, TabularRecord

from tapas.generators import ReturnRaw


# Set some parameters
num_train_samples = 100
num_test_samples = 1000
train_data_size = 5
synthetic_data_size = 5

# Set output directory
curr_time = datetime.now().strftime('%d%m_%H%M%S')
save_dir = os.path.join('runs', curr_time)

# Load data
dataset = TabularDataset.read('tapas/tests/data/test_texas')

# Initialise synthetic data generator
sdg_model = ReturnRaw() # instance of generators.Generator

# Set target
# TODO: Should there be a separate sample_row method on Dataset?
target = TabularRecord.from_dataset(dataset.sample(1))


# The threat model describes the attack for this target user.
threat_model = TargetedAuxiliaryDataMIA(target,
                                        dataset,
                                        sdg_model,
                                        aux_data = None,
                                        sample_real_frac = 0.5,
                                        num_training_records = train_data_size,
                                        num_synthetic_records = synthetic_data_size)

# Initialise attack.
# NOTE: the threat_model contains dataset, and thus a dataset description.
classifier = SetReprClassifier(NaiveRep, RFClassifier, dataset.description)
attack = Groundhog(classifier, dataset.description)

# Train attack.
attack.train(threat_model, num_samples=num_train_samples)

# Generate test data (implicitly through .test).
test_labels, predictions = threat_model.test(attack, num_test_samples,
                                             replace_target=True, save_datasets=True)

# We can also define a second attack, which will reuse memoized datasets.
#attack_cd = ClosestDistanceMIA(threat_model)
#attack_cd.train(num_samples = 100)
#test_labels_cd, predictions_cd = threat_model.test(attack_cd, num_test_samples)

# Do something with predictions.
# TODO: have some reporting mechanism to streamline this.
print(f'Accuracy (GH): {np.mean(np.array(predictions) == np.array(test_labels))}')
#print(f'Accuracy (CD): {np.mean(np.array(predictions_cd) == np.array(test_labels_cd))}')