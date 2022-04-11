from datetime import datetime
import os

from privE.attacks.set_classifiers import LRClassifier, SetReprClassifier, NaiveRep
from privE.threat_models.mia import AuxiliaryDataMIA
from privE.attacks import Groundhog, ClosestDistanceAttack
from privE.datasets import TabularDataset

from privE.generative_models import ReturnRaw


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
threat_model = AuxiliaryDataMIA(target, dataset = dataset, generator = sdg_model,
    auxiliary_test_split = 0.9, num_training_samples = train_data_size,
    num_synthetic_samples = synthetic_data_size)

# If we give the threat model to the .train method, we can remove this.
# train_datasets, train_labels = threat_model.generate_training_samples(num_train_samples)

# Initialise attack.
# NOTE: the threat_model contains dataset, and thus a dataset description.
classifier = SetReprClassifier(NaiveRep, LRClassifier, dataset.description)
attack = Groundhog(threat_model, classifier, dataset.description)

# Train attack.
# attack.train(train_datasets, train_labels)
attack.train(num_samples=num_train_samples)

# Generate test data.
test_datasets, test_labels = threat_model.generate_testing_samples(num_test_samples)

# Predict with attack.
predictions = attack.attack(test_datasets)

# We can also define a second attack, which will reuse memoized datasets.
attack_cd = ClosestDistanceAttack(threat_model)
attack_cd.train(num_samples = 100)
predictions_cd = attack.attack(test_datasets)

# Do something with predictions.
# TODO: have some reporting mechanism to streamline this.
print(f'Accuracy (GH): {np.mean(np.array(predictions) == np.array(test_labels))}')
print(f'Accuracy (CD): {np.mean(np.array(predictions_cd) == np.array(test_labels))}')