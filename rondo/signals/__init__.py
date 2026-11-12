"""Signals (Person A): M0 inspection, M1 ingest and ball cleaning, M2 evidence."""

import os
import sys

import rondo  # noqa: F401  (puts the challenge repo and its evaluation/ on sys.path)

# starter/ holds the organisers' ATD and Smart Tagging loaders.
_STARTER = os.path.join(rondo.CHALLENGE, "starter")
if _STARTER not in sys.path:
    sys.path.insert(0, _STARTER)
