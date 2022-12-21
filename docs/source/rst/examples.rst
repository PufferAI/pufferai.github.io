.. image:: /resource/header.png

|

.. role:: python(code)
    :language: python

Quick Demo on Neural MMO
########################

A simple custom binding for Neural MMO demonstrating the most common usage pattern of PufferLib: defining a new environment binding with a custom policy and training with your framework of choice. The source code for our RLlib and CleanRL tests are included later in this tutorial.

.. literalinclude:: ../../../../pufferlib/tests/test_example.py

Standalone Binding
##################

A more realistic example implementing one of the official NLE baselines. This is a standalone binding without the trainer stub at the bottom.

.. literalinclude:: ../../../../pufferlib/pufferlib/registry/nethack.py

RLLib
#####

Example usage of RLlib with PufferLib utilities. Most of these just streamline the existing RLlib API and add checks for common user errors.

.. literalinclude:: ../../../../pufferlib/tests/test_rllib.py

CleanRL
#######

Example usage of CleanRL with PufferLib. Since CleanRL is not an installable library by design, we've copied the Atari LSTM PPO example and modified it as minimally as possible. PufferLib provides a compatible network and a more robust environment vectorization layer. Note that this layer is quite slow currently, so you may want to look into other options if your environment is fast.

.. literalinclude:: ../../../../pufferlib/tests/test_cleanrl.py

Environment Bindings
####################

We provide default bindings for all tested environments. The default model is a one-layer linear network, optionally wrapped in an LSTM: don't expect it to train well, but it should at least run and give you a starting point. Here are all of the binding definitions:

.. literalinclude:: ../../../../pufferlib/pufferlib/registry/registry.py

