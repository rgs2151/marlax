Core concepts
=============

Responsibilities
----------------

``Environment`` defines one world's transitions. ``Model`` represents a function.
``Agent`` routes model parameters and chooses actions. ``Method`` updates learning
state. The runner supplies batch and time axes, random keys, and resets.
The public interface classes are structural protocols; inheritance is optional.

Environment state, model variables, and runner state are explicit PyTrees.
Configuration objects are immutable. No Python file writes or plotting occur
inside JIT. The online runner and TD methods use feed-forward tabular Q policies
with empty carry ``()``.

World and timestep
------------------

``env.reset(key, params)`` and ``env.step(key, state, actions, params)`` both return
``(state, timestep)``. A step never resets. Reset returns zero rewards, false end
flags, and zero metrics. The runner records the true next observation before
resetting completed worlds.

For one world with ``A`` animals and ``U`` actions:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Value
     - Shape
     - Meaning
   * - Actions
     - ``[A]``
     - Integer action indices
   * - Observation
     - ``[A]``
     - Finite observation index per animal
   * - Action mask
     - ``[A, U]``
     - At least one legal action per row
   * - Reward
     - ``[A]``
     - Repeated team reward in the cooperative grid
   * - Terminated / truncated
     - scalar
     - Episode-level flags
   * - Global state
     - environment-specific
     - Privileged features, excluded from agent input

The runner adds a leading world axis ``[B]``; scan statistics add a time axis
``[T]``. The online runner requires a fixed agent count and identical discrete action
spaces.

Cooperative grid
----------------

Logical x points right and y points up. Actions are stay, south, north, west,
and east. Boundary movement clips; animals may occupy the same cell. All animals
must meet at the center to activate a hidden target pair. They then collect by
occupying the same one of those two targets. An active animal reaching another
edge zone ends the episode in failure. Splitting across the selected targets
adds a penalty but does not terminate. Reward components add to the step cost.

The observation index encodes the ordered animal positions and a target code.
Before activation, that code is zero regardless of the hidden target. Each agent
receives the joint index: these are shared observations, not local decentralized
sensing. The 5×5, two-agent environment has 4,375 possible observation indices.

Time limits are external truncations. Learning bootstraps from the final
observation at truncation, but not after natural termination. Both flags can be
true on the last step; termination takes precedence.

Learning methods
----------------

For every occupied ``(parameter slot, observation, action)``, average the batch's
TD errors and apply alpha once. Repeated visits therefore do not multiply the
learning rate. This is a synchronous batch update, not sequential scalar updates.

Q-learning uses the largest legal next Q-value. SARSA uses the sampled next
behavior action and retains that action for the next actual nonterminal step.
Expected SARSA averages under the next epsilon-greedy distribution. Random
exploration samples only legal actions; greedy ties choose the lowest legal index.

The target is ``reward + gamma * (1 - terminated) * next_value``. Truncation ends
the rollout but retains the final bootstrap. SARSA's bootstrap action at a
truncation belongs to the final observation; the action executed after reset is
sampled from the new world. The exploration schedule is linear in total world
transitions, clamped at its final value.

Independent parameters use a table per animal. Shared parameters pool updates
into one table. Sharing is a modeling decision: identical joint observations and
shared parameters may prevent role specialization. Neither setting changes
whether observations themselves are local or global.
