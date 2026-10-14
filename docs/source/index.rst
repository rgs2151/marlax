MARLAX
======

MARLAX is a small JAX library for cooperative multi-agent reinforcement learning.
Define one world, compose an agent and a learning method, train across batched
worlds, and save a frozen policy independently of training state.

The current release supports tabular Q-learning, SARSA, and Expected SARSA,
independent or shared Q-tables, and the cooperative grid environment. Neural
models, model-based planning, and the new JavaScript export format remain planned.
The existing browser gallery and its saved artifacts are unchanged.

.. toctree::
   :maxdepth: 2

   guide
   tutorials
   inventory
   api
   validation

* :ref:`genindex`
* :ref:`modindex`
