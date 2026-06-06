MARLAX
======

MARLAX is a JAX-first framework for cooperative multi-agent reinforcement learning. It
is being built around fast batched worlds, clear algorithms, and gallery-first
inspection of agent behavior.

.. raw:: html

   <section class="splash-gallery">
     <div class="splash-copy">
       <p class="splash-kicker">Cooperative gridworld</p>
       <h2>Watch the world first.</h2>
       <p>
         The front page is the gallery: two stochastic agents coordinate, recover from
         perturbations, and expose whether the world feels learnable.
       </p>
     </div>
     <div class="interactive-grid-card">
       <canvas id="cooperative-grid-demo" width="720" height="720" aria-label="Interactive cooperative gridworld"></canvas>
     </div>
   </section>

Current Direction
-----------------

- model-free and model-based method zoo
- cooperative multi-agent environment zoo
- unified Zarr experiment storage
- fast batched CPU/GPU execution
- browser demos backed by exported policies

Docs
----

.. toctree::
   :maxdepth: 2
   :caption: Build

   getting_started
   gallery
   environments/index
   algorithms/index
   design/index
   api
