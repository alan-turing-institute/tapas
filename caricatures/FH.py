import toolbox

# Fetch the data from disk (as a toolbox.datasets.Dataset object).
my_dataset = toolbox.datasets.load('my_data.csv', 'data_descriptor.json')

user_ids = my_dataset.unique_users()


# Fetch the SDG model from disk.
sdg_model = toolbox.generators.load(
	# The folder sdg_models/minutemen/ has a "run" file.
	'sdg_models/minutemen/',
	args = {'epsilon': 1, 'two_way_marginals': 'all'})

# We could also imagine having some models directly interfaced in the library.
sdg_model = toolbox.generators.MinuteMen(epsilon=1, two_way_marginals='all')


# Choose a target user to attack.
target_user = np.random.choice(my_dataset.unique_users())

# We could also have heuristics as part of the toolbox?
target_user = toolbox.utils.choose_outlier(my_dataset)


# Choose an attack model.
attack_model = toolbox.attack_models.AuxiliaryDatasetMIA(
	target=target_user,
	auxiliary_test_split=0.9,
	training_data_size=10000,
	synthetic_data_size=10000)

# Alternative: worst-case (DP) attack model.
attack_model = toolbox.attack_models.WorstCaseMIA(
	target=target_user,
	training_data_size=10000,
	synthetic_data_size=10000)


# Generate training datasets under that attack model.

attack_manager = toolbox.AttackManager(my_dataset, sdg_model, attack_model)

# Some attacks require "training pairs" (training_dataset, )
attack_manager.generate_training_samples(num_samples=100)


# Now, load some attacks and add them to the manager.
# If an attack does not support an attack model, adding them to
#  the attack_manager will raise an error.

# Some attacks can be part of thee library already.
attack_manager.add_attack(
	'stadler-basic',
	toolbox.attacks.GroundhogAttack(
		features = 'basic'
	)
)

# Attacks could have parameters to change them.
attack_manager.add_attack(
	'stadler-custom',
	toolbox.attacks.GroundhogAttack(
		features = lambda record: np.sum(record),
		model = sklearn.linear_model.LogisticRegression(),
	)
)

# Or maybe they can be loaded from run scripts (?).
attack_manager.add_attack(
	'my custom attack',
	toolbox.attacks.load(
		# This is a folder with a "run" file.
		'attacks/custom/',
		args = {'foo': 'bar'}
	)
)


# Train all the attacks (if needed) using the samples generated.
attack_manager.fit()

# All the attacks have been trained. You can access them
# independently, and call .predict on the attacks.
print(attack_manager.attacks['stadler-basic']._parameters)


# Test the results: first generate some samples.
attack_manager.generate_test_samples(num_samples=100)


#  Then, add reports for the analysis.
attack_manager.add_report(
	'accuracy',
	toolbox.reports.Accuracy()
)

attack_manager.add_report(
	'roc_curves',
	toolbox.reports.ROCCurve(output_file='roc.pdf')
)


# Finally, apply the attacks on all datasets, and compute reports.
attack_manager.test_report()

print(attack_manager.reports['accuracy'])

# > {'stadler-basic': 0.8, 'stadler-custom': 0.85, 'my custom attack': 0.5}




