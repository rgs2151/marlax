Available tools and APIs
============================

MARLAX provides environments, tabular learning methods, training and evaluation
functions, and policy storage. The learning methods use finite-index observations,
discrete actions, and feed-forward Q-tables.

Environment tools
---------------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - API
     - Use
   * - :class:`marlax.envs.coop_grid.CooperativeGrid`
     - Construct a single cooperative world; size and agent count determine table capacity
   * - :class:`marlax.envs.coop_grid.GridParams`
     - Set time limit and reward components; use ``_replace`` for a modified configuration
   * - :class:`marlax.envs.coop_grid.GridState`
     - Hold positions, target ID, activation, and elapsed steps; also construct explicit scenarios
   * - ``env.spec`` / ``env.default_params()``
     - Read static interface dimensions and default numerical task parameters
   * - ``env.reset`` / ``env.step``
     - Initialize or advance a world with explicit randomness and state
   * - ``env.observe`` / ``env.global_state``
     - Separate actor observations from privileged simulator features
   * - ``env.state_id``
     - Encode ordered positions and the revealed target for the grid table
   * - ``env.timestep``
     - Construct a grid timestep from a state and events
   * - ``env.all_starts``
     - Enumerate ordered starting positions and target IDs for exhaustive evaluation
   * - ``env.center`` / ``env.zones`` / ``env.targets``
     - Inspect grid geometry

Actions are ordered stay, south, north, west, east. The finite table has
``7 * size ** (2 * num_agents)`` observations and grows rapidly with animal count.
``all_starts`` enumerates the corresponding initial cases and is intended for
small worlds. See the first :doc:`tutorial <tutorials>` and :doc:`concepts`.

Models and execution
------------------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - API
     - Use
   * - :class:`marlax.models.tabular.QTable`
     - Represent one finite observation/action table; optionally set optimistic ``initial_value``
   * - ``model.init`` / ``model.apply`` / ``model.initial_carry``
     - Initialize, evaluate, and obtain the empty carry for the Q-table model
   * - :class:`marlax.agents.policy.Agent`
     - Compose the table with independent or shared parameter routing
   * - ``agent.init`` / ``agent.slots`` / ``agent.values``
     - Allocate stacked tables, inspect routing, and obtain per-animal Q-values
   * - ``agent.act`` / ``agent.initial_carry``
     - Sample masked epsilon-greedy behavior with explicit keys and empty carry
   * - :func:`marlax.agents.policy.action_probabilities`
     - Inspect legal epsilon-greedy probabilities; greedy ties use the lowest legal index
   * - :class:`marlax.agents.policy.FrozenPolicy`
     - Store inference variables and choose greedy actions without learning
   * - ``policy.act`` / ``policy.initial_carry``
     - Forward-simulate frozen behavior

Learning methods
--------------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - API
     - Next-state target
   * - :class:`marlax.methods.td.QLearning`
     - Maximum legal next action value
   * - :class:`marlax.methods.td.SARSA`
     - Value of the actual retained next behavior action
   * - :class:`marlax.methods.td.ExpectedSARSA`
     - Expected value under the next epsilon-greedy distribution
   * - ``method.init`` / ``method.update``
     - Initialize learning state and apply averaged batched TD updates
   * - ``method.epsilon``
     - Evaluate the schedule in world transitions
   * - ``method.next_value``
     - Inspect or extend the tabular bootstrap rule
   * - ``method.freeze``
     - Extract a greedy inference policy

All three expose alpha, gamma, epsilon start/end, and epsilon transition budget.
Termination removes the bootstrap; external truncation keeps it. Parameter-sharing
is selected on the agent rather than by a separate algorithm class.

Training and evaluation
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - API
     - Use
   * - :func:`marlax.runners.online.train`
     - Run an additional transition budget, optionally saving a fresh run directory
   * - :func:`marlax.runners.online.initialize`
     - Construct explicit worlds, method state, random key, and first behavior actions
   * - :func:`marlax.runners.online.run_steps`
     - Collect and update through ``scan``; JIT it in a custom experiment loop
   * - :func:`marlax.evaluation.evaluate`
     - Evaluate frozen behavior from seeded resets or supplied initial states
   * - :class:`marlax.runners.online.Run`
     - Access ``policy``, ``state``, ``history``, and ``output``
   * - :class:`marlax.runners.online.TrainState`
     - Retain worlds, timestep, method state, RNG, pending actions, and episode counter
   * - :class:`marlax.methods.td.MethodState`
     - Retain Q variables, transition/update counters, and static agent configuration

Storage
-----------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - API
     - Use
   * - :func:`marlax.storage.save_policy` / :func:`marlax.storage.load_policy`
     - Write/read a standalone greedy table and its JSON manifest
   * - :func:`marlax.storage.save_checkpoint` / :func:`marlax.storage.load_checkpoint`
     - Write/read complete numerical training state using a matching initialized template
   * - :func:`marlax.storage.runtime_metadata`
     - Inspect Python, JAX, NumPy, and visible device details used in run provenance

Inference files are NPZ arrays plus JSON. Checkpoints also require the original
configuration and matching code version. See :doc:`storage`.

Contracts and data records
------------------------------

* :class:`marlax.core.interfaces.Environment`: structural single-world interface.
* :class:`marlax.core.interfaces.Model`: explicit initialization and application.
* :class:`marlax.core.interfaces.Method`: explicit learning state and freezing.
* :class:`marlax.core.data.EnvSpec`: agent count, finite observation count, action count, schema ID.
* :class:`marlax.core.data.PolicyInput`: observation and legal-action mask.
* :class:`marlax.core.data.TimeStep`: policy input, global features, reward, end flags, and events.
* :class:`marlax.core.data.Transition`: batched current/next inputs, actions, rewards, and end flags.
* :class:`marlax.core.data.Decision`: selected actions, carry, and extra Q-values.

Import high-level functions and contracts with ``import marlax as mx``. Import
implementations from ``marlax.envs``, ``marlax.models``, and ``marlax.methods``.
Implementation helpers such as ``select_worlds`` and ``sample_actions`` are
internal runner utilities, not separate extension contracts.

Repository commands and examples
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Tool
     - Use
   * - ``experiments/coop_grid_td/run.py train``
     - Smoke or demo training for any implemented method
   * - ``... evaluate``
     - Load a saved run policy and evaluate all grid starts
   * - ``... resume``
     - Continue a checkpoint into a new run directory
   * - ``... evaluate-saved``
     - Inspect the included pretrained NPZ without retraining
   * - ``... plot``
     - Create an SVG from saved metrics, with no learning run
   * - ``examples/coordination.py``
     - Define and train a custom one-step cooperative environment
   * - ``docs/execute_notebooks.py``
     - Execute all six notebooks and save their outputs
   * - ``python scripts/check_docs.py docs/build/html``
     - Check saved tutorial outputs, generated pages, downloads, images, and local links
   * - ``.github/workflows/pages.yml``
     - Run CPU checks and documentation builds on pull requests; publish successful main builds to Pages
   * - ``python -m pytest -q``
     - Run the small correctness suite
   * - ``sphinx-build -W --keep-going -b html docs/source docs/build/html``
     - Build guides, rendered notebooks, inventory, and API reference without notebook execution
   * - ``python -m http.server 8000 --directory site``
     - Serve the interactive gallery locally

Installation extras
-----------------------

``dev`` installs pytest; ``docs`` installs Sphinx and notebook rendering;
``tutorials`` installs JupyterLab, the execution kernel/client, notebook format
support, and plotting; ``viz`` installs matplotlib; ``gpu`` selects CUDA 13 JAX.
``all`` includes these tools and CUDA support. For a CPU setup use
``.[dev,docs,tutorials]`` rather than ``all``.
