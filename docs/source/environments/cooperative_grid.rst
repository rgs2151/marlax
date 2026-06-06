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

Why This World Exists
---------------------

This environment is small enough for tabular methods but expressive enough to expose
coordination failures. It is the first place to check whether algorithms learn movement,
activation, and shared reward collection.

Gallery Asset
-------------

.. image:: ../_static/gallery/cooperative_grid.gif
   :alt: Cooperative gridworld animation with two mouse-shaped agents.
   :class: gallery-gif
