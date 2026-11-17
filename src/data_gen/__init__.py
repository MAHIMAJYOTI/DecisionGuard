"""Synthetic raw data for DecisionGuard (Phase 1).

A bank's fraud and credit systems do not start from a clean table. They start
from extracts: customers, a credit bureau file, loan applications, card
transactions, and the later performance of loans that were actually booked.

`generate_dataset` builds those extracts. The random seed is fixed, so the
same command always rebuilds the same files. A small number of bad rows are
then written on purpose (nulls, duplicates, impossible dates, negative
amounts) because the later pipeline has to catch them.
"""

from src.data_gen.dataset import generate_dataset

__all__ = ["generate_dataset"]
