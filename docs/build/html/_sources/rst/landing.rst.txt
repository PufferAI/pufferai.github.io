.. image:: /resource/header.png

|

.. role:: python(code)
    :language: python


**Quick links:** `Github <https://github.com/pufferai/pufferlib>`_ | `Discord <https://discord.gg/spT4huaGYV>`_ | `Twitter <https://twitter.com/jsuarez5341>`_

Introduction
############

**WIP -- first stable release of PufferLib should be out by Christmas**

You have an environment, a PyTorch model, and an RL framework that are designed to work together but don't. PufferLib is a wrapper layer that provide better compatibility between `Gym <https://github.com/openai/gym>`_ / `PettingZoo <https://pettingzoo.farama.org>`_ environments and standard reinforcement learning frameworks. You write a native PyTorch network and a short binding for your environment; PufferLib takes care of the rest.

We currently support the following frameworks:
    - `CleanRL <https://github.com/vwxyzjn/cleanrl>`_ - Simple single-file PPO implementation suited for 80% of academic research
    - `RLLib <https://docs.ray.io/en/latest/rllib/index.html>`_ - Industry-grade reinforcement learning library with more features and corresponding overhead

We plan to add additional bindings in the future. These mainly provide a wrapper utility that creates a framework-compliant network from a raw PyTorch model. If you decide to write one of these yourself, please consider opening a pull request to contribute it to the library.

PufferLib is currently tested against the following environments and environment platforms:
    - `Atari (ALE) <https://github.com/mgbellemare/Arcade-Learning-Environment>`_ - Beam Rider, Breakout, Enduro, Pong, Qbert, Seaquest, Space Invaders. Includes option to test against all ALE environments.
    - `Box2D (Gym) <https://www.gymlibrary.dev/environments/box2d/>`_ - Cart Pole
    - `Butterfly (PettingZoo) <https://pettingzoo.farama.org/environments/butterfly/>`_ - Knights Archers Zombies, Cooperative Pong
    - `Griddly <https://github.com/Bam4d/Griddly>`_ - Spiders
    - `MAgent <https://github.com/geek-ai/MAgent>`_ - Default configuration
    - `Gym MicroRTS <https://github.com/Farama-Foundation/MicroRTS-Py>`_ - Default configuration
    - `Nethack (NLE) <https://github.com/facebookresearch/nle>`_
    - `Neural MMO <https://neuralmmo.github.io>`_ - Default configuration

You can add bindings to new environments in only a few lines of code. We encourage you to contribute these to our test cases, as this helps us improve the stability of the library.

Installation
############

The base library is a minimal installation. We provide several optional extras:

.. code-block:: python
   
   pip install pufferlib
   pip install pufferlib[rllib] # Compatible Ray/RLlib versions
   pip install pufferlib[docs] #Build docs locally

   pip install pufferlib[tests] # All test environments
   pip install pufferlib[atari,box2d,butterfly,magent,microrts,nethack,nmmo] # Individual environments

Some of the extra environments have additional dependencies not installable through pip. For easy access to all of the testing environments, use the `PufferTank <https://github.com/pufferai/puffertank>`_ Docker. We suggest this setup for contributing to PufferLib.

Support
#######

We have started a community `Discord <https://discord.gg/spT4huaGYV>`_ for development and support. If you are using PufferLib in conjunction with Neural MMO, note that I run both projects and would be happy to help get you set up.

License
#######

PufferLib is FOSS under the MIT license. This is the full set of tools maintained by PufferAI; we do not have private repositories with additional utilities.
