.. image:: /resource/header.png

|

.. role:: python(code)
    :language: python

Environment Registry
####################

All of our builtin bindings for test environments. Yes, it is this easy.

.. literalinclude:: ../../../../pufferlib/pufferlib/registry/registry.py

Extended Atari Binding
######################

Some of our bindings for common benchmarks include better preprocessing and default policies. This is the default binding for Atari environments.

.. literalinclude:: ../../../../pufferlib/pufferlib/registry/atari.py

Minimal CleanRL Demo
####################

Neural MMO on a minimally modified CleanRL. PufferLib provides a compatible network and a more robust environment vectorization layer. Use as an introductory reference

.. literalinclude:: ../../../../pufferlib/cleanrl_minimal.py

Custom CleanRL Demo
###################

Neural MMO on our customized version of CleanRL. Includes better logging, double-buffering, and streamlined multiagent support. Use as a template for your projects.

.. literalinclude:: ../../../../pufferlib/cleanrl_ppo_lstm.py


RLLib Demo
##########

Neural MMO on RLlib with PufferLib emulation. Significantly more stable than running Neural MMO natively due to limited RLlib test coverage for complex combinations of environment features.

.. literalinclude:: ../../../../pufferlib/rllib_ppo_lstm.py

