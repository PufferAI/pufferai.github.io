.. image:: /resource/header.png

|

This is the current core API. Advanced users can check the PufferLib source for additional tools, but note that these are not yet as well tested.

Emulation
#########

This is the core feature of PufferLib - bind your environment for compatibility with RL libraries

.. automodule:: pufferlib.emulation
   :members:
   :undoc-members:
   :noindex:

Registry
########

Builtin bindings and policies for test environments


Atari
*****

.. automodule:: pufferlib.registry.atari
   :members:
   :undoc-members:
   :noindex:


Butterfly
*********

.. automodule:: pufferlib.registry.butterfly
   :members:
   :undoc-members:
   :noindex:


Classic Control
***************

.. automodule:: pufferlib.registry.classic_control
   :members:
   :undoc-members:
   :noindex:


Griddly
*******

.. automodule:: pufferlib.registry.griddly
   :members:
   :undoc-members:
   :noindex:


MAgent
******

.. automodule:: pufferlib.registry.magent
   :members:
   :undoc-members:
   :noindex:


MicroRTS
********

.. automodule:: pufferlib.registry.microrts
   :members:
   :undoc-members:
   :noindex:


NetHack
*******

.. automodule:: pufferlib.registry.nethack
   :members:
   :undoc-members:
   :noindex:


Neural MMO
**********

.. automodule:: pufferlib.registry.nmmo
   :members:
   :undoc-members:
   :noindex:


Starcraft Multiagent Challenge
******************************

.. automodule:: pufferlib.registry.smac
   :members:
   :undoc-members:
   :noindex:


Models
######

PufferLib model API and default policies

.. automodule:: pufferlib.models
   :members:
   :undoc-members:
   :noindex:


Vectorization
#############

Distributed backends for PufferLib-wrapped environments

.. autoclass:: pufferlib.vectorization.serial.VecEnv
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: pufferlib.vectorization.multiprocessing.VecEnv
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: pufferlib.vectorization.ray.VecEnv
   :members:
   :undoc-members:
   :noindex:

CleanRL Integration 
###################

Wraps your policies for use with CleanRL

.. automodule:: pufferlib.frameworks.cleanrl
   :members:
   :undoc-members:
   :noindex:

RLlib Binding
#############

Wraps your policies for use with RLlib (WIP)

.. automodule:: pufferlib.frameworks.rllib
   :members:
   :undoc-members:
   :noindex: