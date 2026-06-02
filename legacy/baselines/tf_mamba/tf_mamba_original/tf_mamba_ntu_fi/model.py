"""TF-Mamba model for NTU-Fi dataset.

  XH: (Batch, 250, 171)  — time-domain Haar coefficients
  XV: (Batch, 171, 250)  — frequency-domain Haar coefficients

Re-exports TFMamba from shared base; all components defined in
baselines/base_models/tf_mamba_base/model.py.
"""
from baselines.base_models.tf_mamba_base.model import TFMamba

__all__ = ['TFMamba']
