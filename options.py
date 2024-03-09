import argparse

def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="index for GPU, -1 for CPU")

    args = parser.parse_args()
    return args

