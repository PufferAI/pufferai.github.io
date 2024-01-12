.. image:: /resource/header.png

|

Our current public API. Advanced users can check the PufferLib source for additional utilities, but note that we tend to move these around more often. Contributions welcome!

Emulation
#########

Wrap your environments for broad compatibility. Supports passing creator functions, classes, or env objects. The API of the returned PufferEnv is the same as Gym/PettingZoo.

.. autoclass:: pufferlib.emulation.GymnasiumPufferEnv
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: pufferlib.emulation.PettingZooPufferEnv
   :members:
   :undoc-members:
   :noindex:

Environments
############

All included environments expose make_env and env_creator functions. make_env is the one that you want most of the time. The other one is used to expose e.g. class interfaces for environments that support them so that you can pass around static references.

Additionally, all environments expose a Policy class with a baseline model. Note that not all environments have *custom* policies, and the default simply flattens observations before applying a linear layer. Atari, Procgen, Neural MMO, Nethack/Minihack, and Pokemon Red currently have reasonable policies.

The PufferLib Squared environment is used as an example below. Everything is exposed through __init__, so you can call these methods through e.g. pufferlib.environments.squared.make_env

.. autoclass:: pufferlib.environments.ocean.squared.Squared
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: pufferlib.environments.ocean.torch.Policy
   :members:
   :undoc-members:
   :noindex:

Models
######

PufferLib model default policies and optional API. These are not required to use PufferLib.

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

Wrap your PyTorch policies for use with CleanRL but add an LSTM. This requires you to use our policy API. It's pretty simple -- see the default policies for examples.

.. autoclass:: pufferlib.frameworks.cleanrl.RecurrentPolicy
   :members:
   :undoc-members:
   :noindex:

SB3 Binding
###########

Minimal CNN + LSTM example included in demo.py

RLlib Binding
#############

Wrap your policies for use with RLlib (Shelved until RLlib is more stable)

.. automodule:: pufferlib.frameworks.rllib
   :members:
   :undoc-members:
   :noindex:
