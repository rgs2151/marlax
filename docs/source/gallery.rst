Gallery
=======

The gallery is where environment behavior becomes visible. Each entry should be generated
from a script in ``visualize/`` so the site stays reproducible.

Cooperative Gridworld
---------------------

Two agents meet at the center to activate a shared target, then move together to collect
the reward.

.. image:: _static/gallery/cooperative_grid.gif
   :alt: Two mouse-shaped agents cooperating in a gridworld.
   :class: gallery-gif

Asset script
------------

.. code-block:: bash

   conda run -n marlax python visualize/cooperative_grid.py
