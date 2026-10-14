Extending MARLAX
================

Add an environment
------------------

The runnable ``examples/coordination.py`` defines a one-step two-animal game
outside the package and trains it with the same public runner and Expected SARSA:

.. code-block:: bash

   JAX_PLATFORMS=cpu python examples/coordination.py

Implement the ``Environment`` protocol with an immutable configuration class and
a NamedTuple or registered dataclass for dynamic state. ``reset`` and ``step``
return state plus ``TimeStep``; ``observe`` returns only allowed actor information.
``global_state`` provides privileged features separately. Do not put a batch
size, renderer, or reset-on-done behavior inside ``step``.

For the TD runner, ``EnvSpec`` describes finite observation indices and
a shared discrete action space. ``PolicyInput.obs`` is an integer vector per
world. Metrics are a fixed-key dictionary of scalar event values. Use ordinary JAX
array operations so the runner can ``vmap`` and ``scan`` the world. Pass reward
parameters explicitly. A new environment can use the same runner and methods
without editing their source.

``CooperativeGrid.all_starts`` is an environment-specific convenience, not part
of the mandatory contract. Other worlds can provide a batch of initial states or
use seeded reset episodes in ``evaluate``. Its ``max_steps`` is an explicit
static evaluation budget; set it to the desired rollout length.

Add a model
-----------

``Model.init(key, input_spec, output_spec)`` initializes one parameter slot.
``apply(variables, inputs, carry, key)`` returns outputs and carry. ``QTable``
implements a finite observation/action lookup. The agent stacks parameter slots
and applies the model per animal. Custom representations can satisfy the model protocol, but the
implemented TD methods update table entries directly and require QTable-shaped
variables. A neural model requires a corresponding gradient-based learning
method; implementing ``Model`` alone does not make it compatible with these TD
updates.

``QTable(initial_value=0.5)`` provides optimistic initialization without another
model class. Shared and independent parameter routing are agent choices rather
than separate table implementations.

Add a learning method
---------------------

For another tabular bootstrap rule, subclass ``QLearning`` and implement
``next_value(values, batch, epsilon)``. This reuses averaged updates, counters,
and freezing. A new method with different learned state implements
``init``, ``update``, and ``freeze`` directly, and uses a runner appropriate to its
data ordering. The online runner applies immediate tabular updates and retains next
behavior actions for SARSA subclasses.
