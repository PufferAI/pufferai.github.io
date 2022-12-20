.. image:: /resource/header.png

|

This is the current core API. Advanced users can check the PufferLib source for additional tools, but note that these are not yet as well tested.

Emulation
#########

This is the core PufferLib wrapper layer that improves your environment's compatibility with RL libraries.

.. automodule:: pufferlib.emulation
   :members:
   :undoc-members:
   :noindex:

Binding
#######

Utilities for creating a PufferLib binding for a specific environment. The automatic binding should work for most environments. 

.. automodule:: pufferlib.bindings.base
   :members:
   :undoc-members:
   :noindex:

PyTorch Spec
############

.. autoclass:: pufferlib.frameworks.BasePolicy
   :members:
   :noindex:

CleanRL Binding
###############

Allows you to wrap your environment and policy for use with CleanRL

.. automodule:: pufferlib.cleanrl
   :members:
   :undoc-members:
   :noindex:

RLlib Binding
#############

Allows you to wrap your environment and policy for use with RLlib

.. automodule:: pufferlib.rllib
   :members:
   :undoc-members:
   :noindex: