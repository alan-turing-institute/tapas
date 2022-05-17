"""
Primary command-line access script into the prive library
"""

import argparse
from datetime import datetime
import json
import os

import numpy as np

from prive.attacks import load_attack
from prive.datasets import TabularDataset, TabularRecord
from prive.generators import GeneratorFromExecutable, Raw
from prive.threat_models import TargetedAuxiliaryDataMIA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--genpath', '-G', type=str, help='Path to generator executable')
    parser.add_argument('--datapath', '-D', type=str,
                        help='Path to directory containing both csv and json of the test data')
    parser.add_argument('--auxdatapath', '-A', type=str, default=None,
                        help='Path to directory containing both csv and json of the auxiliary data, defaults to none')
    parser.add_argument('--runconfig', '-C', type=str, default='prive/configs/config.json',
                        help='Path to config json file')
    parser.add_argument('--outdir', '-O', type=str, default='runs',
                        help='Output directory to save results to')

    args = parser.parse_args()

    # Set output directory
    curr_time = datetime.now().strftime('%d%m_%H%M%S')
    save_dir = os.path.join(args.outdir, curr_time)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load data
    dataset = TabularDataset.read(args.datapath)
    if args.auxdatapath:
        aux_data = TabularDataset.read(args.auxdatapath)
    else:
        aux_data = None

    # Initialise synthetic data generator
    # TODO: Implement generator_from_executable (probably as a function)
    sdg_model = GeneratorFromExecutable(args.genpath) # instance of generators.Generator

    # Load runconfig
    with open(args.runconfig) as f:
        runconfig = json.load(f)

    # Set target
    # TODO: Should there be a separate sample_row method on Dataset?
    target = TabularRecord.from_dataset(dataset.sample(1))
    print(f'Row {target.id} selected as target') # TODO: Logger

    # The threat model describes the attack for this target user.
    threat_model = TargetedAuxiliaryDataMIA(target,
                                            dataset,
                                            sdg_model,
                                            aux_data = aux_data,
                                            sample_real_frac = runconfig['test_as_aux_frac'],
                                            num_training_records = runconfig['train_data_size'],
                                            num_synthetic_records = runconfig['synthetic_data_size'])

    # Initialise attacks.
    attacks = {}
    for atk in runconfig['attacks']:
        # TODO: Implement load_attack
        attacks[atk['name']] = load_attack(**atk, dataset=dataset)

    # Train attacks.
    for attack in attacks.values():
        attack.train(threat_model, num_samples=runconfig['num_train_samples'])

    # Generate test data (implicitly through .test).
    results = {}
    for atk_name, attack in attacks.items():
        test_labels, predictions = threat_model.test(
            attack, runconfig['num_test_samples'], replace_target=True, save_datasets=True)
        results[atk_name] = {'test_labels': test_labels, 'predictions': predictions}

    # Save the results
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # We can also define a second attack, which will reuse memoized datasets.
    #attack_cd = ClosestDistanceAttack(threat_model)
    #attack_cd.train(num_samples = 100)
    #test_labels_cd, predictions_cd = threat_model.test(attack_cd, num_test_samples)

    # Do something with predictions.
    # TODO: have some reporting mechanism to streamline this.
    for name, res in results.items():
        print(f'Accuracy ({name}): {np.mean(np.array(res["predictions"]) == np.array(res["test_labels"]))}')
    #print(f'Accuracy (CD): {np.mean(np.array(predictions_cd) == np.array(test_labels_cd))}')

if __name__ == '__main__':
    main()