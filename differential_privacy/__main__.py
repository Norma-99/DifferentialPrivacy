import argparse
from differential_privacy.controller import Controller


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    Controller(args).run()



