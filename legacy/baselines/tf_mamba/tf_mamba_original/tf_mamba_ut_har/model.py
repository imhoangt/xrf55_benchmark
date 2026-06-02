"""TF-Mamba model for UT-HAR dataset.

  XH: (Batch, 125, 45)  — time-domain Haar coefficients
  XV: (Batch, 125, 45)  — frequency-domain Haar coefficients

Re-exports TFMamba from shared base; all components defined in
baselines/base_models/tf_mamba_base/model.py.
"""
from baselines.base_models.tf_mamba_base.model import TFMamba

__all__ = ['TFMamba']
