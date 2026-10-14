API tutorials
=================

Six notebooks introduce environment configuration, agents, learning methods,
training, storage, and custom environments. Each includes executable examples
and saved outputs. Follow them in order or choose a topic below.

.. toctree::
   :maxdepth: 1

   Environment and timestep <notebooks/01_environment>
   Models, agents, and action selection <notebooks/02_models_agents>
   Q-learning, SARSA, and Expected SARSA <notebooks/03_learning_methods>
   Training loops and saved-policy evaluation <notebooks/04_training_evaluation>
   Saving policies and resuming training <notebooks/05_checkpoints>
   Building a custom environment <notebooks/06_custom_environment>

What each tutorial teaches
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 46 30

   * - Tutorial
     - Workflow
     - Inspect the result
   * - Environment
     - Configure rewards, reset, step, activate, collect, truncate, and batch
     - State and observation shapes, final and reset states
   * - Models and agents
     - Initialize tables, select sharing, apply masks, sample, and freeze
     - Parameter shapes and action probabilities
   * - Learning methods
     - Compute each TD target and select a method in the training API
     - Worked TD updates and training runs
   * - Training and evaluation
     - Control JIT chunks, load the pretrained table, evaluate, and roll out
     - All 3,750 grid scenarios and a trajectory plot
   * - Checkpoints
     - Save an inference policy, restore full state, and continue SARSA
     - Equality with uninterrupted training
   * - Custom environment
     - Define a one-step cooperative game and reuse existing components
     - Learned coordination with shared parameters

Run locally
---------------

From the repository root, in the environment where you want to run the notebooks:

.. code-block:: bash

   pip install -e ".[docs,tutorials]"
   python -m ipykernel install --user --name marlax --display-name "MARLAX"
   python -m jupyterlab docs/source/notebooks

Choose the MARLAX kernel and run a notebook from top to bottom. The notebook
working directory should be ``docs/source/notebooks``; the saved-policy tutorial
uses that location to find the example policy in ``stores/``.

Every notebook selects CPU before importing JAX. To use a GPU, edit that first
cell before starting a fresh kernel and launch Jupyter with the desired
``CUDA_VISIBLE_DEVICES`` setting. No GPU is required for these tutorials.
Restart the kernel before changing the backend; changing an environment variable
after JAX initializes will not move existing arrays.

The learning-method tutorial uses 8,192-transition runs to demonstrate method
selection. For full training, use the ``demo`` profile in :doc:`experiments`.
The training-and-evaluation tutorial evaluates the included pretrained policy.
Checkpoint examples use temporary directories and clean them up at the end.

To execute all notebooks and refresh their saved outputs:

.. code-block:: bash

   python docs/execute_notebooks.py
   JAX_PLATFORMS=cpu sphinx-build -W --keep-going -b html docs/source docs/build/html

The execution script starts a fresh kernel using its own Python interpreter for
each notebook.

Downloads
-------------

* :download:`Environment and timestep <notebooks/01_environment.ipynb>`
* :download:`Models and agents <notebooks/02_models_agents.ipynb>`
* :download:`Learning methods <notebooks/03_learning_methods.ipynb>`
* :download:`Training and evaluation <notebooks/04_training_evaluation.ipynb>`
* :download:`Checkpoints <notebooks/05_checkpoints.ipynb>`
* :download:`Custom environment <notebooks/06_custom_environment.ipynb>`

Keep downloaded notebooks in the same repository-relative directory when running
the saved-policy tutorial, or explicitly edit its ``root`` path.
