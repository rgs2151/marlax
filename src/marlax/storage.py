from dataclasses import asdict
from pathlib import Path
import json
import platform

import jax
import jax.numpy as jnp
import numpy as np

from marlax.agents import Agent, FrozenPolicy
from marlax.models import QTable


def save_checkpoint(path, state):
    """Save numerical PyTree leaves; restore with a matching initialized template."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    leaves = jax.tree.leaves(state)
    np.savez_compressed(path / "state.npz", **{f"leaf_{i}": np.asarray(x) for i, x in enumerate(leaves)})
    (path / "manifest.json").write_text(json.dumps({"format": "marlax.checkpoint", "version": 1, "leaves": len(leaves)}, indent=2) + "\n")


def load_checkpoint(path, template):
    """Restore state using the original configuration's PyTree structure."""
    leaves, structure = jax.tree.flatten(template)
    with np.load(Path(path) / "state.npz", allow_pickle=False) as arrays:
        restored = [jnp.asarray(arrays[f"leaf_{i}"]) for i in range(len(leaves))]
    return jax.tree.unflatten(structure, restored)


def save_policy(path, policy, spec):
    """Save a tabular inference policy as portable NPZ arrays and JSON metadata."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "variables.npz", q=np.asarray(policy.variables))
    metadata = {
        "format": "marlax.policy", "version": 1, "kind": "q_table",
        "agent": asdict(policy.agent), "environment_spec": asdict(spec),
        "shape": list(policy.variables.shape), "dtype": str(policy.variables.dtype),
        "execution": "greedy", "tie_break": "lowest_legal_action_index",
    }
    (path / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_policy(path):
    """Load a MARLAX tabular policy without its training state."""
    path = Path(path)
    metadata = json.loads((path / "manifest.json").read_text())
    agent = Agent(QTable(**metadata["agent"]["model"]), sharing=metadata["agent"]["sharing"])
    with np.load(path / "variables.npz", allow_pickle=False) as arrays:
        variables = jnp.asarray(arrays["q"])
    return FrozenPolicy(agent, variables)


def runtime_metadata():
    return {
        "python": platform.python_version(), "jax": jax.__version__, "numpy": np.__version__,
        "devices": [str(device) for device in jax.devices()],
    }
