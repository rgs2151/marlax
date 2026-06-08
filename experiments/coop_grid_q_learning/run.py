import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from config import SCENARIOS
from evaluate import evaluate_policy, rollout_scenarios
from export import write_outputs
from plot import (
    write_failure_plot,
    write_learning_plot,
    write_policy_plot,
    write_q_margin_plot,
    write_showcase_rollout_plot,
)
from train import train_policy


def main():
    env, q, history = train_policy()
    final_metrics = evaluate_policy(q, env, show_progress=True)
    scenario_results = rollout_scenarios(q, env, SCENARIOS, show_progress=True)
    write_learning_plot(history)
    write_failure_plot(final_metrics)
    write_policy_plot(q, env, active=False)
    write_policy_plot(q, env, active=True)
    write_q_margin_plot(q)
    write_showcase_rollout_plot(env, scenario_results)
    write_outputs(q, env, history, final_metrics, scenario_results)


if __name__ == "__main__":
    main()
