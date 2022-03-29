## Option 1

```
python3 linkage_cli.py -RC /path/to/attack_config.json -D /path/to/data -G /path/to/generative_model.py -GRC /path/to/generative_model_config.json
```

or for a set of generative models

```
python3 linkage_cli.py -RC /path/to/attack_config.json -D /path/to/data -GL /path/to/gen_model_dir -N list_of_gen_model_names_to_run -GRC /path/to/generative_model_config.json
```
where gen_model_dir is organised as
```
.
└── gen_model_dir
    ├── gen_model_1
    │	├── config.json
    │	├── gen_model.py
    │	└── utils
    └── gen_model_2
        ├── config.json
        ├── gen_model.py
        └── utils
```

## Option 2

As a python package without CLI

```
import toolbox
from generative_models import GenerativeModel
from utils import load_data, fancy_plot

save_dir = 'path/to/save/at'

gen_config = json.load('/path/to/gen/config.json')

data, metadata = load_data('/path/to/data')

gen_model = GenerativeModel(gen_config, metadata)

attack = toolbox.Attack1(generative_model, data, attack_config)

results = attack.evaluate()

fancy_plot(results, save_dir)
```
