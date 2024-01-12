.. role:: python(code)
    :language: python

|

You have an environment, a PyTorch model, and a reinforcement learning library that are designed to work together but don't. PufferLib provides one-line wrappers that make them play nice.

.. card::
  :link: https://colab.research.google.com/drive/142tl_9MiEDXX-E5-6kjwZsOmRYPcFrFU?usp=sharing
  :width: 75%
  :margin: 4 2 auto auto
  :text-align: center

  **Click to Demo PufferLib in Colab**

|
.. raw:: html

    <center>
      <video width=100% height="auto" nocontrols autoplay playsinline muted loop>
        <source src="../_static/banner.webm" type="video/webm">
        <source src="../_static/banner.mp4" type="video/mp4">
        Your browser does not support this video.
      </video>
    </center>

.. raw:: html

    <div style="text-align: center;">
        <div style="display: flex; align-items: center; justify-content: center; margin: auto;">
            <div style="flex-shrink: 0; width: 60px; margin-right: 20px;">
                <a href="https://github.com/pufferai/pufferlib" target="_blank">
                    <img src="https://img.shields.io/github/stars/pufferai/pufferlib?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star pufferai/pufferlib" width="60px">
                </a>
            </div>
            <a href="https://discord.gg/puffer" target="_blank" style="margin-right: 20px;">
                <img src="https://dcbadge.vercel.app/api/server/puffer?style=plastic" alt="Discord">
            </a>
            <a href="https://twitter.com/jsuarez5341" target="_blank">
                <img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez5341" alt="Twitter">
            </a>
        </div>
    </div>

|

Join our community Discord for support and Discussion, follow my Twitter for news, and star the repo to feed the puffer. We also have a :download:`Whitepaper <../_static/neurips_2023_aloe.pdf>` featured at the NeurIPS 2023 ALOE workshop.

.. dropdown:: Installation

  .. tab-set::
    
    .. tab-item:: PufferTank

      `PufferTank <https://github.com/pufferai/puffertank>`_ is a GPU container with PufferLib and dependencies for all environments in the registry, including some that are slow and tricky to install.

      If you have not used containers before and just want everything to work, clone the repository and open it in VSCode. You will need to install the Dev Container plugin as well as Docker Desktop. VSCode will then detect the settings in .devcontainer and set up the container for you.

    .. tab-item:: Pip

      PufferLib is also available as a standard pip package.

      .. code-block:: python
        
        pip install pufferlib

      To install additional environments and frameworks:

      .. code-block:: python
        
        pip install pufferlib[nmmo,cleanrl]

      Note that some environments require additional non-pip dependencies. Follow the additional setup from the maintainers of that environment, or just use PufferTank.
         
.. dropdown:: Contributors

   **Joseph Suarez**: Creator and developer of PufferLib

   **David Bloomin**: 0.4 policy pool/store/selector

   **Nick Jenkins**: Layout for the system architecture diagram. Adversary.design.

   **Andranik Tigranyan**: Streamline and animate the pufferfish. Hire him on UpWork if you like what you see here.

   **Sara Earle**: Original pufferfish model. Hire her on UpWork if you like what you see here.

**You can open this guide in a Colab notebook by clicking the demo button at the top of this page**

Complex environments may have heirarchical observations and actions, variable numbers of agents, and other quirks that make them difficult to work with and incompatible with standard reinforcement learning libraries. PufferLib's emulation layer makes every environment look like it has flat observations/actions and a constant number of agents. Here's how it works with NetHack and Neural MMO, two notoriously complex environments.

.. code-block:: python

  import pufferlib.emulation
  import pufferlib.wrappers

  import nle, nmmo

  def nmmo_creator():
      env = nmmo.Env()
      env = pufferlib.wrappers.PettingZooTruncatedWrapper(env)
      return pufferlib.emulation.PettingZooPufferEnv(env=env)

  def nethack_creator():
      return pufferlib.emulation.GymnasiumPufferEnv(env_creator=nle.env.NLE)

The wrappers give you back a Gymnasium/PettingZoo compliant environment. There is no loss of generality and no change to the underlying environment. You can wrap environments by class, creator function, or object, with or without additional arguments. These wrappers enable us to make some optimizations to vectorization code that would be difficult to implement otherwise. You can choose from a variety of vectorization backends. They all share the same interface with synchronous and asynchronous options.

.. code-block:: python

  import pufferlib.vectorization

  vec = pufferlib.vectorization.Serial
  # vec = pufferlib.vectorization.Multiprocessing
  # vec = pufferlib.vectorization.Ray

  # Vectorization API. Specify total number of environments and number per worker
  # Setting env_pool=True can be much faster but requires some tweaks to learning code
  envs = vec(nmmo_creator, num_envs=4, envs_per_worker=2, env_pool=False)

  # Synchronous API - reset/step
  # obs = envs.reset()[0]

  # Asynchronous API - async_reset, send/recv
  envs.async_reset()
  obs = envs.recv()[0]

Our backends support asynchronous on-policy sampling through a Python implementation of EnvPool. This makes them *faster* than the implementations that ship with most RL libraries. We suggest Serial for debugging and Multiprocessing for most training runs. Ray is a good option if you need to scale beyond a single machine.

PufferLib allows you to write vanilla PyTorch policies and use them with multiple learning libraries. We take care of the details of converting between the different APIs. Here's a policy that will work with *any* environment, with a one-line wrapper for CleanRL.

.. code-block:: python

  import torch
  from torch import nn
  import numpy as np

  import pufferlib.frameworks.cleanrl

  class Policy(nn.Module):
      def __init__(self, env):
          super().__init__()
          self.encoder = nn.Linear(np.prod(
              envs.single_observation_space.shape), 128)
          self.decoders = nn.ModuleList([nn.Linear(128, n)
              for n in envs.single_action_space.nvec])
          self.value_head = nn.Linear(128, 1)

      def forward(self, env_outputs):
          env_outputs = env_outputs.reshape(env_outputs.shape[0], -1)
          hidden = self.encoder(env_outputs)
          actions = [dec(hidden) for dec in self.decoders]
          value = self.value_head(hidden)
          return actions, value

  obs = torch.Tensor(obs)
  policy = Policy(envs.driver_env)
  cleanrl_policy = pufferlib.frameworks.cleanrl.Policy(policy)
  actions = cleanrl_policy.get_action_and_value(obs)[0].numpy()
  obs, rewards, terminals, truncateds, infos, env_id, mask = envs.step(actions)
  envs.close()

There's also an optional policy base class for PufferLib. It just breaks the forward pass into an encode and decode step, which allows us to handle recurrance for you. So far, the code above is fully general and does not rely on PufferLib support for specific environments. For convenience, we also provide environment hooks with standard wrappers and baseline models. Here's a complete example.

.. code-block:: python

  import torch

  import pufferlib.models
  import pufferlib.vectorization
  import pufferlib.frameworks.cleanrl
  import pufferlib.environments.nmmo

  envs = pufferlib.vectorization.Multiprocessing(
      env_creator=pufferlib.environments.nmmo.make_env,
      num_envs=4, envs_per_worker=2)

  policy = pufferlib.environments.nmmo.Policy(envs.driver_env)
  cleanrl_policy = pufferlib.frameworks.cleanrl.Policy(policy)

  env_outputs = envs.reset()[0]
  obs = torch.Tensor(env_outputs)
  actions = cleanrl_policy.get_action_and_value(obs)[0].numpy()
  obs, rewards, terminals, truncateds, infos, env_id, mask = envs.step(actions)
  envs.close()

It's that simple -- almost. If you have an environment with structured observations, you'll have to unpack them in the network forward pass since PufferLib will flatten them in emulation. We provide a utility for this.

.. code-block:: python

  obs = pufferlib.emulation.unpack_batched_obs(
      env_outputs,
      envs.driver_env.flat_observation_space,
      envs.driver_env.flat_observation_structure
  )

That's all you need to get started. The PufferLib repository contains full-length CleanRL scripts with PufferLib integration. SB3 and other integrations coming soon!

Libraries
#########

PufferLib's emulation layer adheres to the Gym and PettingZoo APIs: you can use it with *any* environment and learning library (subject to Limitations). The libraries and environments below are just the ones we've tested. We also provide additional tools to make them easier to work with.

PufferLib provides *pufferlib.frameworks* for the the learning libraries below. These are short wrappers over your vanilla PyTorch policy that handles learning library API details for you. Additionally, if you use our *optional* model API, which just requires you to split your *forward* function into an *encode* and *decode* portion, we can handle recurrance for you. This is the approach we use in our default policies.

.. raw:: html

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/vwxyzjn/cleanrl" target="_blank">
                <img src="https://img.shields.io/github/stars/vwxyzjn/cleanrl?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star CleanRL" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/vwxyzjn/cleanrl">CleanRL</a> provides single-file RL implementations suited for 80+% of academic research. It was designed for simple environments like Atari, but with PufferLib, you can use it with just about anything.</p>
        </div>
    </div>

.. card::
  :link: https://colab.research.google.com/drive/1OMcaJnCAF1UiCJxKIxSS-RdZTuonItYT?usp=sharing
  :width: 75%
  :margin: 4 2 auto auto
  :text-align: center

  **Click to Demo PufferLib + CleanRL in Colab**

Or view it on GitHub `here <https://github.com/PufferAI/PufferLib/blob/experimental/cleanrl_ppo_atari.py>`_

PufferLib also includes a heavily customized version of CleanRL PPO with support for recurrent and non-recurrent models, async environment execution, variable agent populations, self-play, and experiment management. This is the version we use for our research and the NeurIPS 2023 Neural MMO Competition. You can try it out `here <https://github.com/PufferAI/PufferLib/blob/experimental/clean_pufferl.py>`_ 

.. raw:: html

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/anyscale/ray" target="_blank">
                <img src="https://img.shields.io/github/stars/ray-project/ray?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star Ray" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://docs.ray.io/">Ray</a> is a general purpose distributed computing framework that includes <a href="https://docs.ray.io/en/latest/rllib">RLlib</a>, an industry reinforcement learning library.</p>
        </div>
    </div>

We have previously supported RLLib and may again in the future. RLlib has not received updates in a while, and the current release is very buggy. We will update this if the situation improves.

Environments
############

PufferLib ships with Ocean, our first-party testing suite. We also provide integrations for many environments out of the box. Non-pip dependencies are already set up for you in PufferTank. Several environments also include reasonable baseline policies. Join our Discord if you would like to add setup and tests for new environments or improvements to any of the baselines.


.. raw:: html

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/openai/gym" target="_blank">
                <img src="https://img.shields.io/github/stars/openai/gym?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star OpenAI Gym" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/openai/gym">OpenAI Gym</a> is the standard API for single-agent reinforcement learning environments. It also contains some built-in environments. We include <a href="https://www.gymlibrary.dev/environments/box2d/">Box2D</a> in our registry.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/PWhiddy/PokemonRedExperiments" target="_blank">
                <img src="https://img.shields.io/github/stars/PWhiddy/PokemonRedExperiments?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star Pokemon Red" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/PWhiddy/PokemonRedExperiments">Pokemon Red</a> is one of the original Pokemon games for gameboy. This project uses the game as an environment for reinforcement learning. We are actively supporting development on this one!</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/Farama-Foundation/PettingZoo" target="_blank">
                <img src="https://img.shields.io/github/stars/Farama-Foundation/PettingZoo?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star PettingZoo" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://pettingzoo.farama.org">PettingZoo</a> is the standard API for multi-agent reinforcement learning environments. It also contains some built-in environments. We include <a href="https://pettingzoo.farama.org/environments/butterfly/">Butterfly</a> in our registry.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/Farama-Foundation/Arcade-Learning-Environment" target="_blank">
                <img src="https://img.shields.io/github/stars/Farama-Foundation/Arcade-Learning-Environment?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star Arcade Learning Environment" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/Farama-Foundation/Arcade-Learning-Environment">Arcade Learning Environment</a> provides a Gym interface for classic Atari games. This is the most popular benchmark for reinforcement learning algorithms.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/Farama-Foundation/Minigrid" target="_blank">
                <img src="https://img.shields.io/github/stars/Farama-Foundation/Minigrid?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star Minigrid" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/Farama-Foundation/Minigrid">Minigrid</a> is a 2D grid-world environment engine and a collection of builtin environments. The target is flexible and computationally efficient RL research.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/geek-ai/MAgent" target="_blank">
                <img src="https://img.shields.io/github/stars/geek-ai/MAgent?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star MAgent" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/geek-ai/MAgent/blob/master/doc/get_started.md">MAgent</a> is a platform for large-scale agent simulation.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/neuralmmo/environment" target="_blank">
                <img src="https://img.shields.io/github/stars/openai/neural-mmo?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star Neural MMO" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://neuralmmo.github.io">Neural MMO</a> is a massively multiagent environment for reinforcement learning. It combines large agent populations with high per-agent complexity and is the most actively maintained (by me) project on this list.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/openai/procgen" target="_blank">
                <img src="https://img.shields.io/github/stars/openai/procgen?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star Procgen" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/openai/procgen">Procgen</a> is a suite of arcade games for reinforcement learning with procedurally generated levels. It is one of the most computationally efficient environments on this list.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/facebookresearch/nle" target="_blank">
                <img src="https://img.shields.io/github/stars/facebookresearch/nle?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star NLE" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/facebookresearch/nle">Nethack Learning Environment</a> is a port of the classic game NetHack to the Gym API. It combines extreme complexity with high simulation efficiency.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/facebookresearch/minihack" target="_blank">
                <img src="https://img.shields.io/github/stars/facebookresearch/minihack?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star MiniHack" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/facebookresearch/nle">MiniHack Learning Environment</a> is a stripped down version of NetHack with support for level editing and custom procedural generation.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/danijar/crafter" target="_blank">
                <img src="https://img.shields.io/github/stars/danijar/crafter?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star Crafter" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/danijar/crafter">Crafter</a> is a top-down 2D Minecraft clone for RL research. It provides pixel observations and relatively long time horizons.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/Bam4d/Griddly" target="_blank">
                <img src="https://img.shields.io/github/stars/Bam4d/Griddly?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star Griddly" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://griddly.readthedocs.io/en/latest/">Griddly</a> is an extremely optimized platform for building reinforcement learning environments. It also includes a large suite of built-in environments.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/Farama-Foundation/MicroRTS-Py" target="_blank">
                <img src="https://img.shields.io/github/stars/Farama-Foundation/MicroRTS-Py?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star MicroRTS-Py" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/Farama-Foundation/MicroRTS-Py">Gym MicroRTS</a> is a real time strategy engine for reinforcement learning research. The Java configuration is a bit finicky -- we're still debugging this.</p>
        </div>
    </div>

Current Limitations
###################

- No continuous action spaces (WIP)
- Support for heterogenous observations and actions requires you to specify teams such that each team has the same observation and action space. There's no good way around this.

License
#######

PufferLib is free and open-source software under the MIT license. This is the full set of tools maintained by PufferAI. Dev branches are public and we do not have private repositories with additional utilities.
