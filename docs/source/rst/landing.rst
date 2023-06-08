.. role:: python(code)
    :language: python

.. raw:: html

    <center>
      <video width=100% height="auto" nocontrols autoplay playsinline muted loop>
        <source src="../_static/banner.webm" type="video/webm">
        <source src="../_static/banner.mp4" type="video/mp4">
        Your browser does not support this video.
      </video>
    </center>

**Quick links:** `Github <https://github.com/pufferai/pufferlib>`_ | `Baselines <https://api.wandb.ai/links/jsuarez/wue9qkr0>`_ | `Discord <https://discord.gg/spT4huaGYV>`_ | `Twitter <https://twitter.com/jsuarez5341>`_

You have an environment, a PyTorch model, and a reinforcement learning framework that are designed to work together but don't. PufferLib is a wrapper layer that makes RL on complex game environments as simple as RL on Atari. You write a native PyTorch network and a short binding for your environment; PufferLib takes care of the rest.

Join our community `Discord <https://discord.gg/spT4huaGYV>`_ to get support or if you are interested in contributing to the project.

| :ref:`Minimal CleanRL Demo` Neural MMO on minimally modified CleanRL. Use as an introductory reference.
| :ref:`Custom CleanRL Demo` Neural MMO on our customized version of CleanRL. Use as a template for your projects.

.. literalinclude:: ../../../../pufferlib/tests/test_docs_nmmo.py

Installation
############

**Docker Setup:** `PufferTank <https://github.com/pufferai/puffertank>`_ ships with PufferLib and all test environments. Includes a GPU-enabled VSCode Dev Container config file for easy local development. We highly recommend this setup for PufferLib contributors.

**Pip Install:** Requires PyTorch (we tested on 1.12.0+cu116). Does not include non-pip environment dependencies.

.. code-block:: python
   
   pip install pufferlib
   pip install pufferlib[rllib] # Compatible Ray/RLlib versions
   pip install pufferlib[docs] # Build docs locally

   pip install pufferlib[tests] # All test environments
   pip install pufferlib[atari,box2d,butterfly,magent,microrts,nethack,nmmo] # Individual environments


Included frameworks and environments
####################################

Frameworks are supported by a 50-150 line wrapper that formats a native PyTorch policy for compatibility with the given model API. We currently support the following frameworks:

| `CleanRL <https://github.com/vwxyzjn/cleanrl>`_ - Simple single-file PPO implementation suited for 80% of academic research
| `RLLib <https://docs.ray.io/en/latest/rllib/index.html>`_ - Industry-grade reinforcement learning library with more features and corresponding overhead

Environments are supported by a 1-line call that wraps the provided class or environment creator in a PufferEnv. PufferLib is compatible with both `Gym <https://github.com/openai/gym>`_ and `PettingZoo <https://pettingzoo.farama.org>`_ environments and includes bindings for the following projects:

| `Atari (ALE) <https://github.com/mgbellemare/Arcade-Learning-Environment>`_ - Beam Rider, Breakout, Enduro, Pong, Qbert, Seaquest, Space Invaders. Includes option to test against all ALE environments.
| `Box2D (Gym) <https://www.gymlibrary.dev/environments/box2d/>`_ - Cart Pole
| `Butterfly (PettingZoo) <https://pettingzoo.farama.org/environments/butterfly/>`_ - Knights Archers Zombies, Cooperative Pong
| `Griddly <https://github.com/Bam4d/Griddly>`_ - Spiders
| `MAgent <https://github.com/geek-ai/MAgent>`_ - Default configuration
| `Gym MicroRTS <https://github.com/Farama-Foundation/MicroRTS-Py>`_ - Default configuration
| `Nethack (NLE) <https://github.com/facebookresearch/nle>`_
| `Neural MMO <https://neuralmmo.github.io>`_ - Default configuration


Authorship
##########

| `Joseph Suarez: <https://people.csail.mit.edu/jsuarez>`_ Creator of PufferLib. If you are using PufferLib in conjunction with `Neural MMO <https://neuralmmo.github.io>`_, note that I run both projects and would be happy to help get you set up.


Contributing
############

.. dropdown:: List of Contributors

   **Joseph Suarez**: Creator and developer of PufferLib

   **Nick Jenkins**: Layout for the system architecture diagram. Adversary.design.

   **Andranik Tigranyan**: Streamline and animate the pufferfish. Hire him on UpWork if you like what you see here.

   **Sara Earle**: Original pufferfish model. Hire her on UpWork if you like what you see here.
 
We welcome contributions from the community. Please communicate with us on `Discord <https://discord.gg/spT4huaGYV>`_ before opening an issue or pull request on Github. We are particularly interested in contributions to the following areas:

| **Framework bindings:** We currently support CleanRL and RLLib. We would like to add bindings for other frameworks such as Stable Baselines and Tianshou.
| **Environment bindings:** We would like to improve our test coverage by adding bindings for additional environments. This also requires specifying any external dependencies in PufferTank.
| **Testing:** PufferLib is a stability-crucial library. We would like to add better coverage to our test suite. 
| **Performance:** We would like to improve the performance of our vectorized environments. Our current implementation is a bottleneck for many projects.
| **Baselines:** We would like to add more rigerous performance and regression evaluations to out test suite.
| **Documentation:** We would like to improve our documentation and examples.


License
#######

PufferLib is free and open-source software under the MIT license. This is the full set of tools maintained by PufferAI; we do not have private repositories with additional utilities.