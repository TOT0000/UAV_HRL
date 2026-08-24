"""Deprecated thin wrapper for the canonical controlled-DQN experiment."""

import sys
from run_experiment import main

CANONICAL_METHOD = "td3_dinkelbach_dqn"

def train():
    return main([CANONICAL_METHOD, *sys.argv[1:]])

if __name__ == "__main__":
    raise SystemExit(train())
