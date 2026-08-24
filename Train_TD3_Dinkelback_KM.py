"""Deprecated thin wrapper for the canonical KM+TD3 experiment."""

import sys
from run_experiment import main

CANONICAL_METHOD = "km_td3_dinkelbach"

def train():
    return main([CANONICAL_METHOD, *sys.argv[1:]])

if __name__ == "__main__":
    raise SystemExit(train())
