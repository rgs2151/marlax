Cooperative Gridworld
=====================

The first world is a cooperative grid task inspired by the earlier MARLAX prototype.

Behavior
--------

- Two agents start in separate grid cells.
- They meet at the center to activate the reward.
- A reward target appears on one grid edge.
- Both agents move to the active target to collect together.
- The gallery animation loops through several starts and targets.
- The browser demo lets agents be dragged and then recover under their policy.

Why This World Exists
---------------------

This environment is small enough for tabular methods but expressive enough to expose
coordination failures. It is the first place to check whether algorithms learn movement,
activation, and shared reward collection.

Interactive Demo
----------------

The homepage canvas runs in the browser. It uses a value-driven stochastic policy with
small per-agent trait variation so agents are not identical clones, then lets pointer
dragging perturb the state.

Policy Variation
----------------

The demo samples persistent per-agent traits instead of giving each mouse a fixed
script. This follows the spirit of `parameter-space noise
<https://arxiv.org/abs/1706.01905>`_: a rollout keeps a coherent behavioral style while
still exploring. The softmax action sampler is a small maximum-entropy style choice,
and randomized starts, targets, and traits keep the world closer to domain
randomization.

Gallery Asset
-------------

.. image:: ../_static/gallery/cooperative_grid.gif
   :alt: Cooperative gridworld animation with two mouse-shaped agents.
   :class: gallery-gif
