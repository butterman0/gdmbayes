# Examples

Worked notebooks live in the [`examples/`](https://github.com/butterman0/gdmbayes/tree/main/examples)
directory of the repository. They are not executed as part of the documentation build;
open them on GitHub or run them locally with Jupyter.

| Notebook | What it covers |
|---|---|
| [`basic_usage.ipynb`](https://github.com/butterman0/gdmbayes/blob/main/examples/basic_usage.ipynb) | Fit the frequentist {class}`~gdmbayes.GDM` and the Bayesian {class}`~gdmbayes.spGDMM` to a small synthetic dataset, inspect sampling diagnostics, and predict dissimilarities for new sites. |
| [`model_variants.ipynb`](https://github.com/butterman0/gdmbayes/blob/main/examples/model_variants.ipynb) | Configure variance structures and spatial effects with the string / callable API of {class}`~gdmbayes.ModelConfig`. Builds models without sampling. |
| [`southwest_running_example.ipynb`](https://github.com/butterman0/gdmbayes/blob/main/examples/southwest_running_example.ipynb) | Complete workflow on the Southwest Australia dataset (94 sites, 865 plant species): frequentist and Bayesian fits, cross-validation, and biological-space maps. |
