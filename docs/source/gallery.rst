Gallery
=======

The gallery is where environment behavior becomes visible. Each entry is generated from
a script in ``visualize/`` so the site stays reproducible.

Cooperative Gridworld
---------------------

Two agents run several cooperation trials. Each trial starts from a different pair of
positions, activates the center, reveals a new target, and collects together.

.. image:: _static/gallery/cooperative_grid.gif
   :alt: Two mouse-shaped agents cooperating in a gridworld.
   :class: gallery-gif

Asset script
------------

.. code-block:: bash

   conda run -n marlax python visualize/cooperative_grid.py
