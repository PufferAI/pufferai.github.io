.. image:: /resource/header.png

|

Our current public API. Advanced users can check the PufferLib source for additional utilities, but note that we tend to move these around more often. Contributions welcome!

Emulation
#########

Wrap your environments for broad compatibility. Supports passing creator functions, classes, or env objects. The API of the returned PufferEnv is the same as Gym/PettingZoo.

.. autoclass:: pufferlib.emulation.GymPufferEnv
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: pufferlib.emulation.PettingZooPufferEnv
   :members:
   :undoc-members:
   :noindex:

Registry
########

make_env functions and policies for included environments.

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

Crafter
*******

.. automodule:: pufferlib.registry.crafter
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

Procgen
*******

.. automodule:: pufferlib.registry.procgen
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

.. autoclass:: pufferlib.vectorization.Serial
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: pufferlib.vectorization.Multiprocessing
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: pufferlib.vectorization.Ray
   :members:
   :undoc-members:
   :noindex:

CleanRL Integration 
###################

Wrap your PyTorch policies for use with CleanRL

.. autoclass:: pufferlib.frameworks.cleanrl.Policy
   :members:
   :undoc-members:
   :noindex:

Recurrence requires you to subclass our base policy instead. See the default policies for examples.

.. autoclass:: pufferlib.frameworks.cleanrl.RecurrentPolicy
   :members:
   :undoc-members:
   :noindex:

RLlib Binding
#############

Wrap your policies for use with RLlib (WIP)

.. automodule:: pufferlib.frameworks.rllib
   :members:
   :undoc-members:
   :noindex: