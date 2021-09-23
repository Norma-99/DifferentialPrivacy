# Differential Privacy in a fog-embedded framework


## Execution commands

To execute the whole project

Note: the --config option must be a .json file with all the selected presets for the execution

```bash
python3 -m differential_privacy --config sample_config.json
```

To create a new NeuralNetwork to test use the utils.py file 

Note: you can add some parameters (--hidden_count, --layer_size)

```bash
python3 scripts/utils.py --hidden_count 3 --layer_size 32 32 32 > test_net.json
```

## Project folders description

Privacy-preserving framework for fog computing environments. 
Moreover, we evaluate the framework in two different datasets. 
Both datasets are analyzed through a Feed Forward Neural Network (FFNN). 
To validate the results we compare the proposed framework with a traditional centralized architecture.
