Unified Storage
===============

The storage target is one ``.zarr`` run directory per experiment run.

Why Zarr
--------

- It stores chunked N-dimensional arrays.
- It supports hierarchical groups and metadata.
- It works for long batched training traces.
- It can be read by analysis code without loading everything.

Run layout
----------

.. code-block:: text

   runs/
     <run_id>.zarr/
       attrs
       config/
       env/
       learner/
       steps/
       episodes/
       checkpoints/
       metrics/

Groups
------

- ``attrs``: schema version, creation time, package version, git commit, host metadata.
- ``config``: serialized env, learner, storage, and training configs.
- ``env``: static environment metadata such as grid size, targets, action names.
- ``learner``: static learner metadata such as method name and hyperparameters.
- ``steps``: dense step arrays with leading dimensions ``time x env``.
- ``episodes``: episode boundary arrays and derived episode summaries.
- ``checkpoints``: learner snapshots, including Q-tables or model params.
- ``metrics``: low-cardinality scalar summaries for dashboards and figures.

Step arrays
-----------

- ``state_id``: integer encoded states before action.
- ``next_state_id``: integer encoded states after transition.
- ``actions``: ``time x env x agent`` integer actions.
- ``rewards``: ``time x env x agent`` reward values.
- ``done``: ``time x env`` episode termination flags.
- ``collected``: ``time x env`` task success flags.
- ``epsilon``: ``time`` exploration schedule.
- ``positions``: optional ``time x env x agent x 2`` grid positions.

Chunking
--------

- Chunk primarily along time.
- Keep ``env`` and ``agent`` dimensions inside each chunk.
- Store small metadata in attrs, not separate arrays.
- Keep checkpoint chunks independent from step logs.

First implementation
--------------------

- Add a ``RunStore`` that creates the group structure.
- Write fixed-size step buffers to Zarr.
- Add one Q-table checkpoint writer.
- Add one reader that computes episode success rate from storage.
