from datetime import datetime
import os

from prive.attacks.set_classifiers import LRClassifier, SetReprClassifier, NaiveRep
from prive.threat_models.mia import TargetedAuxiliaryDataMIA
from prive.attacks import Groundhog, ClosestDistanceAttack
from prive.datasets import TabularDataset

from prive.generators import ReturnRaw


# Set some parameters
num_train_samples = 20
num_test_samples = 10
train_data_size = 1000
synthetic_data_size = 1000

# Set output directory
curr_time = datetime.now().strftime('%d%m_%H%M%S')
save_dir = os.path.join('runs', curr_time)

# Load data
dataset = TabularDataset.read('data/texas')

# Initialise synthetic data generator
sdg_model = ReturnRaw() # instance of generators.Generator

# Set target
# TODO: What should target be? DF? Dataset instance?
# Should there be a separate sample_row method on Dataset?
target = dataset.sample(1)


# The threat model describes the attack for this target user.
# TODO: this is probably not the right name here, come to think of it.
threat_model = TargetedAuxiliaryDataMIA(target,
                                        dataset,
                                        sdg_model,
                                        aux_data = None,
                                        sample_real_frac = 0.5,
                                        num_training_records = train_data_size,
                                        num_synthetic_records = synthetic_data_size)

# If we give the threat model to the .train method, we can remove this.
# train_datasets, train_labels = threat_model.generate_training_samples(num_train_samples)

# Initialise attack.
# NOTE: the threat_model contains dataset, and thus a dataset description.
classifier = SetReprClassifier(NaiveRep, LRClassifier, dataset.description)
attack = Groundhog(threat_model, classifier, dataset.description)

# Train attack.
# attack.train(train_datasets, train_labels)
attack.train(num_samples=num_train_samples)

# Generate test data (implicitly through .test).
test_labels, predictions = threat_model.test(attack, num_test_samples)

# We can also define a second attack, which will reuse memoized datasets.
attack_cd = ClosestDistanceAttack(threat_model)
attack_cd.train(num_samples = 100)
test_labels_cd, predictions_cd = threat_model.test(attack_cd, num_test_samples)

# Do something with predictions.
# TODO: have some reporting mechanism to streamline this.
print(f'Accuracy (GH): {np.mean(np.array(predictions) == np.array(test_labels))}')
print(f'Accuracy (CD): {np.mean(np.array(predictions_cd) == np.array(test_labels_cd))}')