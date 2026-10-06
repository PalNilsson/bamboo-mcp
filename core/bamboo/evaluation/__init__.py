"""Evaluation harnesses for Bamboo's retrieval and planning layers.

Pure-stdlib and import-cheap by design: nothing here is imported on a request
path, and nothing here may pull in an optional dependency, so the harnesses
stay runnable on a bare checkout.
"""
