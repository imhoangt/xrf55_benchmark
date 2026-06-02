"""TF-Mamba model for XRF55 (phase streams).

  XH: (Batch, 500, 90)  — horizontal Haar coefficients transposed
  XV: (Batch, 90, 500)  — vertical Haar coefficients

Re-exports TFMamba from shared base; all components defined in
baselines/tf_mamba_base/model.py.
"""
from baselines.base_models.tf_mamba_base.model import TFMamba

__all__ = ['TFMamba']
