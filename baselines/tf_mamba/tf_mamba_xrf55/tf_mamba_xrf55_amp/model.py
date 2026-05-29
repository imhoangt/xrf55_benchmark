"""TF-Mamba model for XRF55 (amplitude streams).

  XH: (Batch, 500, 135)  — horizontal Haar coefficients transposed
  XV: (Batch, 135, 500)  — vertical Haar coefficients

Re-exports TFMamba from shared base; all components defined in
baselines/tf_mamba_base/model.py.
"""
from baselines.base_models.tf_mamba_base.model import TFMamba

__all__ = ['TFMamba']
