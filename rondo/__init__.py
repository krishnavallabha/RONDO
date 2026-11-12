"""RONDO -- Robust Ownership from Noisy Data, Offline."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHALLENGE = os.path.join(ROOT, "more-than-a-hack-2027")

# The official scorer and game constants live in the challenge repo; import them, never copy them.
for _path in (CHALLENGE, os.path.join(CHALLENGE, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
