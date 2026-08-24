"""Deprecated thin wrapper for canonical random service assignment."""

import sys
from run_experiment import main

CANONICAL_METHOD = "random_assignment_td3_dinkelbach"

def train():
    return main([CANONICAL_METHOD, *sys.argv[1:]])

if __name__ == "__main__":
    raise SystemExit(train())
