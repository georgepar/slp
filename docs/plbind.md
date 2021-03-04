# Pytorch Lightning Bindings

These bindings help to build multi-purpose LightningModules and LightningDataModules, that can be utilized for many tasks / datasets.

Note, this is not in line with the pytorch-lightning mantra where everything about an experiment should be contained in a single module.

I agree this can help reproducibility, but I find it tedious to always copy and paste or even worse rewrite boilerplate code for metric calculation and implementing hooks.

With the extensive logging and sane configuration management in slp, reproducibility while developing new models is less of an issue.

My current workflow is to use these modules for fast development cycles, and when I need to publish a specific model, I can then copy and paste it into an isolated LightningModule to make it easier for the future reader. This way we can have the best of both worlds


::: slp.plbind.dm
::: slp.plbind.helpers
::: slp.plbind.module
::: slp.plbind.trainer