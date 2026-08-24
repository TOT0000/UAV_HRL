"""Deprecated thin wrapper for the canonical direct-ratio DDPG experiment."""

import sys
from run_experiment import main

CANONICAL_METHOD = "ddpg_ratio"

def train():
    return main([CANONICAL_METHOD, *sys.argv[1:]])

if __name__ == "__main__":
    raise SystemExit(train())
