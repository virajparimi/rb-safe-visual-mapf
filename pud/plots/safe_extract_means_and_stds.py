import numpy as np
from pathlib import Path
from argparse import ArgumentParser


def extract_multi_agent_metrics(
    records, num_agents, search_based=(False, None), skip_indices=[], deltas=[]
):
    success_rate = 0.0
    times = []
    mean_steps = []
    mean_rewards = []
    mean_cumulative_costs = []
    for idx, record in enumerate(records):
        if idx in skip_indices:
            continue
        steps = []
        rewards = []
        successes = []
        cumulative_costs = []
        if len(record[0]) == 0:
            if search_based[0]:
                for i in range(num_agents):
                    successes.append(search_based[1][idx][i]["success"])
                all_success = all(successes)
                if all_success:
                    success_rate += 1
                    for i in range(num_agents):
                        steps.append(search_based[1][idx][i]["steps"])
                        rewards.append(search_based[1][idx][i]["rewards"])
                        cumulative_costs.append(
                            search_based[1][idx][i]["cumulative_costs"]
                        )

                    mean_steps.append(np.mean(steps))
                    mean_rewards.append(np.mean(rewards))
                    mean_cumulative_costs.append(np.max(cumulative_costs))
            continue
        for i in range(num_agents):
            successes.append(record[i]["success"])
        all_success = all(successes)
        if all_success:
            cc = 0
            for i in range(num_agents):
                cc += record[i]["cumulative_costs"]
            if cc <= deltas[idx]:
                success_rate += 1
                for i in range(num_agents):
                    steps.append(record[i]["steps"])
                    rewards.append(record[i]["rewards"])
                    cumulative_costs.append(record[i]["cumulative_costs"])
                times.append(record[-1]["total_time"])

                mean_steps.append(np.mean(steps))
                mean_rewards.append(np.mean(rewards))
                mean_cumulative_costs.append(np.max(cumulative_costs))

    metrics = {
        "times": times,
        "mean_steps": mean_steps,
        "mean_rewards": mean_rewards,
        "mean_cumulative_costs": mean_cumulative_costs,
        "success_rate": success_rate / (len(records) - len(skip_indices)),
    }

    return metrics


def means_and_stddevs(metrics, num_agents):
    cc_values = {}
    for metric_name, metric in metrics.items():
        cc_values[metric_name] = (
            np.mean(metric["mean_cumulative_costs"]),
            np.std(metric["mean_cumulative_costs"]),
            metric["success_rate"],
        )

    print(f"{'Num Agents':<40} : {num_agents:>27}")
    for metric_name, (mean, stddev, success_rate) in cc_values.items():
        value_str = f"{mean:.2f} ± {stddev:.2f} ({success_rate:.2f} %)"
        print(f"{metric_name:<40} : {value_str:>27}")


def collect_metrics(basedir, problem_type, n_agents, fallback=True):

    multi_agent_basedir = basedir / problem_type

    methods = [
        "unconstrained",
        "unconstrained_reward_search",
        "constrained",
        "constrained_reward_search",
        "constrained_risk_search",
        "full_constrained_risk_search",
        "lagrangian_search",
        "biobjective_search",
    ]
    rb_methods = [
        "risk_budgeted_search",
        "risk_bounded_uniform_search",
        "risk_bounded_utility_search",
        "risk_bounded_inverse_utility_search",
    ]

    # Load the risk-bound data
    cbs_lb_data = np.load(
        multi_agent_basedir / "cbs_risk_bounds" / f"lb_{n_agents}.npy",
        allow_pickle=True,
    )
    cbs_ub_data = np.load(
        multi_agent_basedir / "cbs_risk_bounds" / f"ub_{n_agents}.npy",
        allow_pickle=True,
    )

    # Determine indices to skip based on invalid bounds.
    skip_indices = []
    for pb_idx in range(len(cbs_lb_data)):
        lb_val = cbs_lb_data[pb_idx]
        ub_val = cbs_ub_data[pb_idx]
        if lb_val == -1 or ub_val == -1 or lb_val > ub_val:
            skip_indices.append(pb_idx)
    print(
        f"Agents {n_agents}: Skipping {len(skip_indices)} out of {len(cbs_lb_data)} problems due to invalid bounds."
    )

    full_deltas = []
    for pct in [0, 0.25, 0.5, 0.75, 1.0]:
        # Compute the delta values (upper bounds) for this risk percentage.
        deltas = []
        for pb_idx in range(len(cbs_lb_data)):
            if pb_idx in skip_indices:
                deltas.append(-1)
                continue
            lb_val = cbs_lb_data[pb_idx]
            ub_val = cbs_ub_data[pb_idx]
            delta = lb_val + pct * (ub_val - lb_val)
            deltas.append(delta)
        full_deltas.append(deltas)

    # Pre-load metrics for unconstrained methods (they are risk-percent independent).
    unrb_records = {}
    for method in methods:
        records = np.load(
            multi_agent_basedir / f"{method}_records_{n_agents}.npy", allow_pickle=True
        )
        unrb_records[method] = records

    # Pre-load risk-bounded records for each risk percentage for rb_methods.
    # We assume each file returns an array-like structure with length 5, one for each
    # risk percentage (0, 0.25, 0.5, 0.75, 1.0).
    rb_records = {}
    for method in rb_methods:
        records = np.load(
            multi_agent_basedir / f"{method}_records_{n_agents}.npy", allow_pickle=True
        )
        rb_records[method] = records

    for idx, pct in enumerate([0, 0.25, 0.5, 0.75, 1.0]):
        metrics = {}
        for method in methods:
            method_records = unrb_records[method]
            method_metrics = extract_multi_agent_metrics(
                method_records,
                n_agents,
                skip_indices=skip_indices,
                deltas=full_deltas[idx],
            )
            metrics[method] = method_metrics

        # For each risk-bounded method, extract the metrics corresponding to this risk percentage.
        for method in rb_methods:
            # Here we select the records corresponding to the current risk percent index.
            method_records = rb_records[method][idx]
            method_metrics = extract_multi_agent_metrics(
                method_records,
                n_agents,
                skip_indices=skip_indices,
                deltas=full_deltas[idx],
            )
            metrics[method] = method_metrics

        # Combine metrics from all methods for plotting.
        # We also include the computed delta values as a dictionary with key "mean_cumulative_costs".
        metrics_list = [metrics[method] for method in methods + rb_methods]
        metrics_list.append({"mean_cumulative_costs": deltas})

        valid_mask = np.array(full_deltas[idx]) != -1
        delta = float(np.mean(np.array(full_deltas[idx])[valid_mask]))

        print("-" * 70)
        print(f"Delta: {delta:.2f} (Risk Percentage: {pct})")
        means_and_stddevs(metrics=metrics, num_agents=n_agents)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--fallback", action="store_true", default=False)

    args = parser.parse_args()

    num_agents = [1, 5, 10, 20]
    problem_types = ["easy", "medium", "hard"]
    env_types = [
        "centerdot",
        "sc2_staging_08",
        "sc0_staging_20",
        "sc3_staging_05",
        "sc3_staging_11",
        "sc3_staging_15",
    ]

    for env_type in env_types:
        print("*" * 50)
        print(f"\tEnvironment Type: {env_type}")
        print("*" * 50)
        basedir = Path("pud/plots/data/" + env_type)
        if "staging" in env_type:
            num_agents = [1, 5, 10]
        for problem_type in problem_types:
            for n_agent in num_agents:
                print("-" * 50)
                print(f"\tProblem Type: {problem_type}")
                print("-" * 50)
                try:
                    collect_metrics(
                        basedir, problem_type, n_agent, fallback=args.fallback
                    )
                except FileNotFoundError:
                    import traceback

                    traceback.print_exc()
                    continue
                print()
            print("\n\n")
