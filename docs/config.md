# Configuration

Utilities for parsing configuration from YAML files and merging them with argparse CLI args (See Getting started for a concrete example).

For this I use OmegaConf and I have extended it with integration for argparse.

This way we can tweak models and run investigative experiments with weird configurations from the CLI, without polluting our configuration files.

When some configuration shows promise then we can create a configuration file out of it with detailed description and set it in stone for reproducibility.

The whole process is transparent, if you follow the conventions.


::: slp.config.config_parser
::: slp.config.nlp
::: slp.config.omegaconf