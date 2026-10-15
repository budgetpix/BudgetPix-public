"""Locate the released pMF PyTorch code (github.com/Lyy-iiis/pMF, branch `torch`) and put it on sys.path.

Set PMF_ROOT to that checkout (the directory holding models/, utils/, pmf.py, evaluate.py). The checkout's package is a
plain `models` folder, so no other `models` package may be on PYTHONPATH at the same time.
"""
import os
import sys


def pmf_root():
    root = os.environ.get("PMF_ROOT", "")
    if not root:
        raise SystemExit("set PMF_ROOT to a checkout of https://github.com/Lyy-iiis/pMF (branch torch); from the repository "
                         "root: git clone -b torch https://github.com/Lyy-iiis/pMF upstream/pMF && export PMF_ROOT=$PWD/upstream/pMF")
    if not os.path.exists(os.path.join(root, "models", "pmfDiT.py")):
        raise SystemExit(f"PMF_ROOT={root} does not contain models/pmfDiT.py -- is it the torch branch checkout?")
    return os.path.abspath(root)


def add_pmf_root():
    r = pmf_root()
    if r not in sys.path:
        sys.path.insert(0, r)
    return r
