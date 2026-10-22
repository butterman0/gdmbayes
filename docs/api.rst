API reference
=============

Everything below is importable from the top-level ``gdmbayes`` namespace.

.. currentmodule:: gdmbayes

Estimators
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GDM
   spGDMM

Preprocessing
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GDMPreprocessor

Configuration
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ModelConfig
   SamplerConfig

Variance and spatial-effect functions
-------------------------------------

Built-in callables selected by :class:`ModelConfig`; custom callables with the
same signature may be supplied instead.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   variance_homogeneous
   variance_covariate_dependent
   variance_polynomial
   spatial_abs_diff
   spatial_squared_diff

.. autosummary::
   :toctree: generated/

   VARIANCE_FUNCTIONS
   SPATIAL_FUNCTIONS

Plotting
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   plot_isplines
   plot_predictor_importance
   plot_obs_vs_pred
   plot_link_curve
   plot_ppc
   crps_boxplot

Maps
----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rgb_biological_space
