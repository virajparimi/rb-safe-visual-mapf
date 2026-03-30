import numpy as np
from pathlib import Path
from argparse import ArgumentParser


def extract_single_agent_metrics(records, search_based=(False, None)):
    success_rate = 0.0
    steps = []
    rewards = []
    cumulative_costs = []
    for idx, record in enumerate(records):
        if len(record) == 0:
            if search_based[0] and search_based[1][idx]["success"]:
                # print("Fallback succeeded")
                success_rate += 1
                steps.append(search_based[1][idx]["steps"])
                rewards.append(search_based[1][idx]["rewards"])
                cumulative_costs.append(search_based[1][idx]["cumulative_costs"])
            # elif search_based[0]:
            #     print("Fallback failed")
            continue
        elif record["success"]:
            success_rate += 1
            steps.append(record["steps"])
            rewards.append(record["rewards"])
            cumulative_costs.append(record["cumulative_costs"])

    metrics = {
        "steps": steps,
        "rewards": rewards,
        "cumulative_costs": cumulative_costs,
        "success_rate": success_rate / len(records),
    }
    return metrics


def extract_multi_agent_metrics(records, num_agents, search_based=(False, None)):
    success_rate = 0.0
    mean_steps = []
    mean_rewards = []
    mean_cumulative_costs = []
    if search_based[0]:
        fallback_num = 0
        fallback_successes = 0
    for idx, record in enumerate(records):
        steps = []
        rewards = []
        successes = []
        cumulative_costs = []
        if len(record[0]) == 0:
            if search_based[0]:
                fallback_num += 1
                for i in range(num_agents):
                    successes.append(search_based[1][idx][i]["success"])
                all_success = all(successes)
                if all_success:
                    fallback_successes += 1
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
                    # mean_cumulative_costs.append(np.mean(cumulative_costs))
            continue
        for i in range(num_agents):
            successes.append(record[i]["success"])
        all_success = all(successes)
        if all_success:
            success_rate += 1
            for i in range(num_agents):
                steps.append(record[i]["steps"])
                rewards.append(record[i]["rewards"])
                cumulative_costs.append(record[i]["cumulative_costs"])

            mean_steps.append(np.mean(steps))
            mean_rewards.append(np.mean(rewards))
            mean_cumulative_costs.append(np.max(cumulative_costs))
            # mean_cumulative_costs.append(np.mean(cumulative_costs))

    # if search_based[0]:
    #     if fallback_num != 0:
    #         ratio = fallback_successes / fallback_num  # type: ignore
    #         print(f"Fallbacks Success Ratio: {ratio:.2f}")
    metrics = {
        "mean_steps": mean_steps,
        "mean_rewards": mean_rewards,
        "mean_cumulative_costs": mean_cumulative_costs,
        "success_rate": success_rate / len(records),
    }

    return metrics


def means_and_stddevs(metrics, num_agents):
    key = "mean_cumulative_costs" if num_agents > 1 else "cumulative_costs"
    unconstrained_cc = metrics["unconstrained"][key]
    unconstrained_search_cc = metrics["unconstrained_search"][key]
    constrained_cc = metrics["constrained"][key]
    constrained_search_cc = [metric[key] for metric in metrics["constrained_search"]]
    constrained_search_uc = [metric[key] for metric in metrics["constrained_search_uc"]]

    if "unconstrained_search_disjoint" in metrics:
        unconstrained_search_ds_cc = metrics["unconstrained_search_disjoint"][key]
        constrained_search_ds_cc = [
            metric[key] for metric in metrics["constrained_search_disjoint"]
        ]
        constrained_search_ds_uc = [
            metric[key] for metric in metrics["constrained_search_disjoint_uc"]
        ]
        constrained_search_risk_bounded_ds = [
            metric[key] for metric in metrics["constrained_search_risk_bounded_disjoint"]
        ]
        constrained_search_risk_bounded_ds_uc = [
            metric[key] for metric in metrics["constrained_search_risk_bounded_disjoint_uc"]
        ]

    unconstrained_cc_mean = np.mean(unconstrained_cc)
    unconstrained_search_cc_mean = np.mean(unconstrained_search_cc)
    constrained_cc_mean = np.mean(constrained_cc)
    constrained_search_cc_means = [np.mean(metric) for metric in constrained_search_cc]
    constrained_search_uc_means = [np.mean(metric) for metric in constrained_search_uc]

    if "unconstrained_search_disjoint" in metrics:
        unconstrained_search_ds_cc_mean = np.mean(unconstrained_search_ds_cc)
        constrained_search_ds_cc_means = [np.mean(metric) for metric in constrained_search_ds_cc]
        constrained_search_ds_uc_means = [np.mean(metric) for metric in constrained_search_ds_uc]
        constrained_search_risk_bounded_ds_means = [
            np.mean(metric) for metric in constrained_search_risk_bounded_ds
        ]
        constrained_search_risk_bounded_ds_uc_means = [
            np.mean(metric) for metric in constrained_search_risk_bounded_ds_uc
        ]

    unconstrained_cc_stddev = np.std(unconstrained_cc)
    unconstrained_search_cc_stddev = np.std(unconstrained_search_cc)
    constrained_cc_stddev = np.std(constrained_cc)
    constrained_search_cc_stddevs = [np.std(metric) for metric in constrained_search_cc]
    constrained_search_uc_stddevs = [np.std(metric) for metric in constrained_search_uc]

    if "unconstrained_search_disjoint" in metrics:
        unconstrained_search_ds_cc_stddev = np.std(unconstrained_search_ds_cc)
        constrained_search_ds_cc_stddevs = [np.std(metric) for metric in constrained_search_ds_cc]
        constrained_search_ds_uc_stddevs = [np.std(metric) for metric in constrained_search_ds_uc]
        constrained_search_risk_bounded_ds_stddevs = [
            np.std(metric) for metric in constrained_search_risk_bounded_ds
        ]
        constrained_search_risk_bounded_ds_uc_stddevs = [
            np.std(metric) for metric in constrained_search_risk_bounded_ds_uc
        ]

    unconstrained_sr = metrics["unconstrained"]["success_rate"]
    unconstrained_search_sr = metrics["unconstrained_search"]["success_rate"]
    constrained_sr = metrics["constrained"]["success_rate"]
    constrained_search_sr = [metric["success_rate"] for metric in metrics["constrained_search"]]
    constrained_search_uc_sr = [metric["success_rate"] for metric in metrics["constrained_search_uc"]]

    if "unconstrained_search_disjoint" in metrics:
        unconstrained_search_ds_sr = metrics["unconstrained_search_disjoint"]["success_rate"]
        constrained_search_ds_sr = [
            metric["success_rate"] for metric in metrics["constrained_search_disjoint"]
        ]
        constrained_search_ds_uc_sr = [
            metric["success_rate"] for metric in metrics["constrained_search_disjoint_uc"]
        ]
        constrained_search_risk_bounded_ds_sr = [
            metric["success_rate"] for metric in metrics["constrained_search_risk_bounded_disjoint"]
        ]
        constrained_search_risk_bounded_ds_uc_sr = [
            metric["success_rate"] for metric in metrics["constrained_search_risk_bounded_disjoint_uc"]
        ]

    edge_cost_factors = [0.1, 0.25, 0.5, 0.75, 1.0]

    print(f"Num Agents\t\t\t: {num_agents}")
    print(
        f"Unconstrained\t\t\t: {unconstrained_cc_mean:.2f} +/- {unconstrained_cc_stddev:.2f} "
        f"({unconstrained_sr:.2f})"
    )
    print(
        f"Unconstrained Search\t\t: {unconstrained_search_cc_mean:.2f} +/- {unconstrained_search_cc_stddev:.2f} "
        f"({unconstrained_search_sr:.2f})"
    )
    if "unconstrained_search_disjoint" in metrics:
        print(
            f"Unconstrained Search Disjoint\t: {unconstrained_search_ds_cc_mean:.2f} +/- "
            f"{unconstrained_search_ds_cc_stddev:.2f} ({unconstrained_search_ds_sr:.2f})"
        )
    print(
        f"Constrained\t\t\t: {constrained_cc_mean:.2f} +/- {constrained_cc_stddev:.2f} ({constrained_sr:.2f})"
    )
    for idx, metric in enumerate(constrained_search_cc_means):
        print(
            f"Constrained Search ({edge_cost_factors[idx]})\t: "
            f"{metric:.2f} +/- {constrained_search_cc_stddevs[idx]:.2f} ({constrained_search_sr[idx]:.2f})"
        )
    for idx, metric in enumerate(constrained_search_uc_means):
        print(f"Constrained Search UC ({edge_cost_factors[idx]})\t: "
              f"{metric:.2f} +/- {constrained_search_uc_stddevs[idx]:.2f} ({constrained_search_uc_sr[idx]:.2f})")

    if "unconstrained_search_disjoint" in metrics:
        for idx, metric in enumerate(constrained_search_ds_cc_means):
            print(
                f"Constrained Search Disjoint ({edge_cost_factors[idx]})\t: "
                f"{metric:.2f} +/- {constrained_search_ds_cc_stddevs[idx]:.2f} ({constrained_search_ds_sr[idx]:.2f})"
            )
        for idx, metric in enumerate(constrained_search_ds_uc_means):
            print(f"Constrained Search Disjoint UC ({edge_cost_factors[idx]})\t: "
                  f"{metric:.2f} +/- {constrained_search_ds_uc_stddevs[idx]:.2f} "
                  f"({constrained_search_ds_uc_sr[idx]:.2f})")
        for idx, metric in enumerate(constrained_search_risk_bounded_ds_means):
            print(f"Constrained Search Risk Bounded Disjoint ({edge_cost_factors[idx]})\t: "
                  f"{metric:.2f} +/- {constrained_search_risk_bounded_ds_stddevs[idx]:.2f} "
                  f"({constrained_search_risk_bounded_ds_sr[idx]:.2f})")
        for idx, metric in enumerate(constrained_search_risk_bounded_ds_uc_means):
            print(f"Constrained Search Risk Bounded Disjoint UC ({edge_cost_factors[idx]})\t: "
                  f"{metric:.2f} +/- {constrained_search_risk_bounded_ds_uc_stddevs[idx]:.2f} "
                  f"({constrained_search_risk_bounded_ds_uc_sr[idx]:.2f})")


def collect_metrics(basedir, problem_type, n_agents, fallback=True):

    if n_agents == 1:
        local_basedir = basedir / "single_agent" / problem_type
        unconstrained_records = np.load(
            local_basedir / "unconstrained_records.npy", allow_pickle=True
        )
        unconstrained_search_records = np.load(
            local_basedir / "unconstrained_search_records.npy", allow_pickle=True
        )
        constrained_records = np.load(
            local_basedir / "constrained_records.npy", allow_pickle=True
        )
        constrained_search_factored_records = np.load(
            local_basedir / "constrained_search_factored_records_uc.npy",
            allow_pickle=True,
        )
        constrained_search_factored_records_cc = np.load(
            local_basedir / "constrained_search_factored_records.npy", allow_pickle=True
        )

        unconstrained_metrics = extract_single_agent_metrics(unconstrained_records)
        unconstrained_search_metrics = extract_single_agent_metrics(
            unconstrained_search_records, (fallback, unconstrained_records)
        )
        constrained_metrics = extract_single_agent_metrics(constrained_records)
        constrained_search_factored_metrics = [
            extract_single_agent_metrics(csr, (fallback, constrained_records))
            for csr in constrained_search_factored_records
        ]
        constrained_search_factored_metrics_cc = [
            extract_single_agent_metrics(csr, (fallback, constrained_records))
            for csr in constrained_search_factored_records_cc
        ]

        return means_and_stddevs(
            {
                "unconstrained": unconstrained_metrics,
                "unconstrained_search": unconstrained_search_metrics,
                "constrained": constrained_metrics,
                "constrained_search_uc": constrained_search_factored_metrics,
                "constrained_search": constrained_search_factored_metrics_cc,
            },
            n_agents,
        )
    else:
        local_basedir = basedir / "multi_agent" / problem_type

        unconstrained_records = np.load(
            local_basedir / f"unconstrained_records_{n_agents}.npy", allow_pickle=True
        )
        unconstrained_search_records = np.load(
            local_basedir / f"unconstrained_search_records_{n_agents}.npy",
            allow_pickle=True,
        )
        # unconstrained_search_ds_records = np.load(
        #     local_basedir / f"unconstrained_search_ds_records_{n_agents}.npy",
        #     allow_pickle=True,
        # )
        constrained_records = np.load(
            local_basedir / f"constrained_records_{n_agents}.npy", allow_pickle=True
        )
        constrained_search_factored_records = np.load(
            local_basedir / f"constrained_search_factored_records_{n_agents}_uc.npy",
            allow_pickle=True,
        )
        # constrained_search_ds_factored_records = np.load(
        #     local_basedir / f"constrained_search_ds_factored_records_{n_agents}_uc.npy",
        #     allow_pickle=True,
        # )
        # constrained_search_risk_bounded_ds_factored_records = np.load(
        #     local_basedir / f"risk_bounded_constrained_search_ds_factored_records_{n_agents}_uc.npy",
        #     allow_pickle=True,
        # )
        constrained_search_factored_records_cc = np.load(
            local_basedir / f"constrained_search_factored_records_{n_agents}.npy",
            allow_pickle=True,
        )
        # constrained_search_ds_factored_records_cc = np.load(
        #     local_basedir / f"constrained_search_ds_factored_records_{n_agents}.npy",
        #     allow_pickle=True,
        # )
        # constrained_search_risk_bounded_ds_factored_records_cc = np.load(
        #     local_basedir / f"risk_bounded_constrained_search_ds_factored_records_{n_agents}.npy",
        #     allow_pickle=True,
        # )

        unconstrained_metrics = extract_multi_agent_metrics(
            unconstrained_records, n_agents
        )
        unconstrained_search_metrics = extract_multi_agent_metrics(
            unconstrained_search_records, n_agents, (fallback, unconstrained_records)
        )
        # unconstrained_search_ds_metrics = extract_multi_agent_metrics(
        #     unconstrained_search_ds_records, n_agents,# (fallback, unconstrained_records)
        # )
        constrained_metrics = extract_multi_agent_metrics(constrained_records, n_agents)
        constrained_search_factored_metrics = [
            extract_multi_agent_metrics(csr, n_agents, (fallback, constrained_records))
            for csr in constrained_search_factored_records
        ]
        # constrained_search_ds_factored_metrics = [
        #     extract_multi_agent_metrics(csr, n_agents,)# (fallback, constrained_records))
        #     for csr in constrained_search_ds_factored_records
        # ]
        # constrained_search_risk_bounded_ds_factored_metrics = [
        #     extract_multi_agent_metrics(csr, n_agents,)# (fallback, constrained_records))
        #     for csr in constrained_search_risk_bounded_ds_factored_records
        # ]
        constrained_search_factored_metrics_cc = [
            extract_multi_agent_metrics(csr, n_agents, (fallback, constrained_records))
            for csr in constrained_search_factored_records_cc
        ]
        # constrained_search_ds_factored_metrics_cc = [
        #     extract_multi_agent_metrics(csr, n_agents,)# (fallback, constrained_records))
        #     for csr in constrained_search_ds_factored_records_cc
        # ]
        # constrained_search_risk_bounded_ds_factored_metrics_cc = [
        #     extract_multi_agent_metrics(csr, n_agents,)# (fallback, constrained_records))
        #     for csr in constrained_search_risk_bounded_ds_factored_records_cc
        # ]

        return means_and_stddevs(
            {
                "unconstrained": unconstrained_metrics,
                "unconstrained_search": unconstrained_search_metrics,
                # "unconstrained_search_disjoint": unconstrained_search_ds_metrics,
                "constrained": constrained_metrics,
                "constrained_search_uc": constrained_search_factored_metrics,
                # "constrained_search_disjoint_uc": constrained_search_ds_factored_metrics,
                "constrained_search": constrained_search_factored_metrics_cc,
                # "constrained_search_disjoint": constrained_search_ds_factored_metrics_cc,
                # "constrained_search_risk_bounded_disjoint_uc": constrained_search_risk_bounded_ds_factored_metrics,
                # "constrained_search_risk_bounded_disjoint": constrained_search_risk_bounded_ds_factored_metrics_cc,
            },
            n_agents,
        )


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
                    collect_metrics(basedir, problem_type, n_agent, fallback=args.fallback)
                except FileNotFoundError:
                    import traceback
                    traceback.print_exc()
                    continue
                print()
            print("\n\n")
