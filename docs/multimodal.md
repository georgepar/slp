# Multimodal Modules

We include strong baselines for multimodal fusion and state-of-the-art paper implementations.

## Fusers

This module contains the implementation of basic fusion algorithms and fusion pipelines.

The fusers are implemented for arbitrary number of input modalities, unless otherwise stated and are geared towards sequential inputs.

A fusion pipeline consists generally of three stages

* Pre-fuse processing: Perform some common operations to all input modalities (e.g. project to a common dimension.)
* Fuser: Fuse all modality representations into a single vector (e.g. concatenate all modality features using CatFuser).
* Timesteps Pooling: Aggregate fused features for all timesteps into a single vector (e.g. add all timesteps with SumPooler)

::: slp.modules.fuse

## Multimodal encoders

These modules implement mid and late fusion. The general structure of a multimodal encoder contains:

* N Unimodal encoders (e.g. RNNs), where N is the number of input modalities
* A fusion pipeline

We furthermore implement Multimodal classifiers, which consist of a multimodal encoder followed by an `nn.Linear` layer.

A special mention should be added for our `MultimodalBaseline`. This baseline consists of RNN encoders followed by an attention fuser and an RNN timesteps poolwer in multimodal tasks and is tuned on CMU-MOSEI. The default configuration is provided through static methods and achieve strong performance.

::: slp.modules.multimodal

## M3

::: slp.modules.mmdrop
::: slp.modules.m3


## Multimodal Feedback

::: slp.modules.feedback
::: slp.modules.mmlatch
