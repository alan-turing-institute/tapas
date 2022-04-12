from datetime import datetime
import os

from prive.attacks.set_classifiers import LRClassifier, SetReprClassifier, NaiveRep
from attack_models import Groundhog
from datasets import TabularDataset

from generative_models import ReturnRaw

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


# Create training datasets
# TODO: Find home
def create_synthetic_from_neighbouring(generator,
                                       dataset,
                                       target,
                                       num_samples,
                                       train_size,
                                       syn_size):
    """
    Generate synthetic datasets from pairs of neighbouring training datasets
    """
    # Split dataset
    datasets = dataset.split(num_splits=num_samples,
                             split_size=train_size,
                             drop=target)

    synthetic_datasets_no_T = []
    synthetic_datasets_with_T = []

    for dataset in datasets_without_T:
        synthetic_datasets_no_T.append(generator(dataset, syn_size))

        dataset.add(target)
        synthetic_datasets_with_T.append(generator(dataset, syn_size))

    labels = ([0] * num_samples) + ([1] * num_samples)
    synthetic_datasets = synthetic_datasets_no_T + synthetic_datasets_with_T

    return synthetic_datasets, labels

train_datasets, train_labels = create_synthetic_from_neighbouring(
    sdg_model, dataset, target, num_train_samples, train_data_size, synthetic_data_size
)

# Initialise attack model
classifier = SetReprClassifier(NaiveRep, LRClassifier, dataset.description)
attack = Groundhog(classifier, dataset.description)

# Train attack model
attack.train(train_datasets, train_labels)

# Generate test data
test_datasets, test_labels = create_synthetic_from_neighbouring(
    sdg_model, dataset, target, num_test_samples, train_data_size, synthetic_data_size
)

# Predict with attack
predictions = attack.attack(test_datasets)

# Do something with predictions
print(f'Accuracy: {np.mean(np.array(predictions) == np.array(test_labels))}')