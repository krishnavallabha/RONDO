"""Harness (C): external matches, their labels, and the official scorer wired to both.

Nothing in here ever reads this match's ground truth. Parameters are chosen on external matches
with provider or event-derived labels, and a change is kept only when it beats the bootstrap
noise band there.
"""

from rondo.harness import datasets, labels  # noqa: F401
