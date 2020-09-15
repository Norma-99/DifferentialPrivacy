import argparse
import importlib
from differential_privacy.utils import read_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--framework', required=True)
    args = parser.parse_args()

    fw = importlib.import_module(f'differential_privacy.frameworks.{args.framework}')
    fw.run(read_config(args.config))


