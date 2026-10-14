import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

import jax.numpy as jnp
import numpy as np

import marlax as mx
from marlax.envs import CooperativeGrid, GridParams
from marlax.methods import ExpectedSARSA, QLearning, SARSA
from marlax.models import QTable


ROOT = Path(__file__).resolve().parents[2]
METHODS = {"q_learning": QLearning, "sarsa": SARSA, "expected_sarsa": ExpectedSARSA}
PROFILES = {"smoke": (32, 1024, 16), "demo": (2048, 12_288_000, 1000)}


def components(config):
    env = CooperativeGrid(**config["env"])
    params = GridParams(**config["params"])
    agent = mx.Agent(QTable(**config["model"]), sharing=config["sharing"])
    method = METHODS[config["method"]](**config["learning"])
    return env, params, agent, method


def evaluate_run(policy, env, params):
    report = mx.evaluate(policy, env, params, initial_states=env.all_starts(), max_steps=params.time_limit)
    report["policy_sha256"] = hashlib.sha256(np.asarray(policy.variables).tobytes()).hexdigest()
    return report


def source_snapshot():
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
    }
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT, text=True).split("\0")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in sorted(set(names)):
            path = ROOT / name
            if path.is_file() and path.suffix in {".py", ".toml", ".yml", ".rst", ".md"}:
                archive.add(path, arcname=name)
    return metadata, buffer.getvalue()


def plot_run(path):
    import matplotlib.pyplot as plt

    history = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    report = json.loads((path / "evaluation.json").read_text())
    output = path / "plots"
    output.mkdir(exist_ok=True)
    with plt.rc_context({"font.family": "serif", "mathtext.fontset": "cm", "savefig.transparent": True}):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), dpi=300)
        axes[0].plot([row["env_steps"] for row in history], [row["reward_per_step"] for row in history], color="darkred", linewidth=1)
        axes[0].set(xlabel="World transitions", ylabel="Mean team reward per step")
        labels = ["Collected", "Wrong target", "Truncated", "Unfinished"]
        rates = [report[name] for name in ("collected_rate", "wrong_target_rate", "truncation_rate", "unfinished_rate")]
        axes[1].bar(labels, rates, color=["darkgreen", "darkred", "midnightblue", "0.5"])
        axes[1].set(ylabel="Fraction of evaluation episodes", ylim=(-0.05, 1.05), yticks=[0, 1])
        axes[1].tick_params(axis="x", labelrotation=25)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["bottom", "left"]].set_position(("outward", 5))
        fig.tight_layout()
        fig.savefig(output / "learning.svg", bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Tabular cooperative-grid experiments; never writes the gallery.")
    parser.add_argument("command", choices=["train", "evaluate", "resume", "evaluate-saved", "plot"])
    parser.add_argument("--method", choices=METHODS, default="q_learning")
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-steps", type=int)
    args = parser.parse_args()

    if args.command == "plot":
        plot_run(args.run)
        return

    if args.command == "evaluate-saved":
        env = CooperativeGrid()
        params = env.default_params()
        with np.load(ROOT / "stores/coop_grid_q_learning/latest/q_values.npz", allow_pickle=False) as arrays:
            policy = mx.FrozenPolicy(mx.Agent(QTable()), jnp.asarray(arrays["q"]))
        print(json.dumps(evaluate_run(policy, env, params), indent=2))
        return

    num_envs, budget, chunk = PROFILES[args.profile]
    if args.env_steps is not None:
        budget = args.env_steps
    config = {
        "env": {"size": 5, "num_agents": 2}, "params": GridParams()._asdict(),
        "model": {"initial_value": 0.0}, "sharing": "independent", "method": args.method,
        "learning": {"alpha": 0.03, "gamma": 0.97, "epsilon_start": 1.0, "epsilon_end": 0.02, "epsilon_steps": budget},
    }
    state = None
    seed = args.seed
    if args.command in ("evaluate", "resume"):
        saved = json.loads((args.run / "config.json").read_text())
        config = saved["experiment"]
        seed, num_envs, chunk = saved["seed"], saved["num_envs"], saved["chunk_steps"]
    env, params, agent, method = components(config)
    if args.command == "evaluate":
        report = evaluate_run(mx.load_policy(args.run / "policy"), env, params)
        (args.run / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    if args.command == "resume":
        template = mx.initialize(env, params, agent, method, seed, num_envs)
        state = mx.load_checkpoint(args.run / "checkpoint", template)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "runs/coop_grid_td" / f"{stamp}-{config['method']}-seed{seed}"
    source_metadata, source_archive = source_snapshot()
    start = time.perf_counter()
    run = mx.train(env, params, agent, method, seed=seed, num_envs=num_envs, env_steps=budget,
                   chunk_steps=chunk, output=output, state=state, config=config)
    report = evaluate_run(run.policy, env, params)
    report["elapsed_train_and_eval_seconds"] = time.perf_counter() - start
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    provenance = json.loads((output / "provenance.json").read_text())
    provenance.update(source_metadata)
    provenance["resumed_from"] = str(args.run) if args.command == "resume" else None
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "source.tar.gz").write_bytes(source_archive)
    print(json.dumps({"run": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
