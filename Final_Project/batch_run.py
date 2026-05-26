import pandas as pd
from mesa.batchrunner import batch_run
from model import SchoolSmokingModel


BASE_PARAMS = {
    "n_students": 300,
    "n_groups": 8,
    "initial_smoking_rate": 0.10,
    "p_in": 0.08,
    "p_out": 0.01,
    "share_naive": 0.40,
    "share_sophisticated": 0.30,

    "peer_influence_weight": 2.0,
    "protective_peer_influence_weight": 0.75,

    "smoking_reinforcement_weight": 0.5,
    "nonsmoking_reinforcement_weight": 0.8,
    "risk_bias_strength": 1.0,
    "smoking_threshold": 1.5,
    "quit_rate": 0.10,

    "information_intervention_step": None,
    "information_intervention_strength": 0.0,

    "peer_intervention_step": None,
    "new_smoking_peer_influence_weight": None,
    "new_protective_peer_influence_weight": None
}


def make_scenarios():
    scenarios = {
        "baseline": {
            "information_intervention_step": None,
            "information_intervention_strength": 0.0,
            "peer_intervention_step": None,
            "new_smoking_peer_influence_weight": None,
            "new_protective_peer_influence_weight": None
        }
    }

    timings = {
        "early": 10,
        "late": 90
    }

    strengths = {
        "weak": 0.25,
        "strong": 0.75,
    }

    for timing_name, step in timings.items():

        scenarios[f"{timing_name}_reduce_smoking_peer"] = {
            "peer_intervention_step": step,
            "new_smoking_peer_influence_weight": 1.0,
            "new_protective_peer_influence_weight": 0.75
        }

        scenarios[f"{timing_name}_strengthen_protective_peer"] = {
            "peer_intervention_step": step,
            "new_smoking_peer_influence_weight": 2.0,
            "new_protective_peer_influence_weight": 2.0
        }

        scenarios[f"{timing_name}_reduce_general_peer"] = {
            "peer_intervention_step": step,
            "new_smoking_peer_influence_weight": 1.0,
            "new_protective_peer_influence_weight": 0.375
        }

        for strength_name, info_strength in strengths.items():

            scenarios[f"{timing_name}_information_only_{strength_name}"] = {
                "information_intervention_step": step,
                "information_intervention_strength": info_strength
            }

            scenarios[f"{timing_name}_combined_protective_{strength_name}"] = {
                "information_intervention_step": step,
                "information_intervention_strength": info_strength,
                "peer_intervention_step": step,
                "new_smoking_peer_influence_weight": 2.0,
                "new_protective_peer_influence_weight": 2.0
            }

            scenarios[f"{timing_name}_combined_reduce_smoking_peer_{strength_name}"] = {
                "information_intervention_step": step,
                "information_intervention_strength": info_strength,
                "peer_intervention_step": step,
                "new_smoking_peer_influence_weight": 1.0,
                "new_protective_peer_influence_weight": 0.75
            }

            scenarios[f"{timing_name}_combined_reduce_general_peer_{strength_name}"] = {
                "information_intervention_step": step,
                "information_intervention_strength": info_strength,
                "peer_intervention_step": step,
                "new_smoking_peer_influence_weight": 1.0,
                "new_protective_peer_influence_weight": 0.375
            }

    return scenarios


SCENARIOS = make_scenarios()


OUTPUTS = [
    "Smoking rate",
    "Naive smoking rate",
    "Sophisticated smoking rate",
    "Time-consistent smoking rate",
    "Average smoking reinforcement",
    "Average non-smoking reinforcement",
    "Average smoking peer exposure",
    "Average protective peer exposure",
    "Average net peer utility",
    "Smoking homophily",
    "Average true risk sensitivity",
    "Average risk perception",
    "Average future discount",
    "Average forecast accuracy",
    "Average perceived health cost",
    "Average expected future addiction cost",
    "Smoking peer influence weight",
    "Protective peer influence weight"
]


def run_scenario(scenario_name, scenario_params, iterations=10, max_steps=180):
    params = BASE_PARAMS.copy()
    params.update(scenario_params)

    results = batch_run(
        SchoolSmokingModel,
        parameters=params,
        rng=[None] * iterations,
        max_steps=max_steps,
        data_collection_period=1,
        number_processes=1,
        display_progress=True
    )

    df = pd.DataFrame(results)
    df["scenario"] = scenario_name

    return df


def get_step_column(df):
    if "Step" in df.columns:
        return "Step"

    if "step" in df.columns:
        return "step"

    raise ValueError("No Step or step column found in batch output.")


def summarize_by_run(df):
    step_col = get_step_column(df)
    run_cols = ["scenario"]

    if "RunId" in df.columns:
        run_cols.append("RunId")

    if "iteration" in df.columns:
        run_cols.append("iteration")

    rows = []

    for keys, run_df in df.groupby(run_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        key_dict = dict(zip(run_cols, keys))

        run_df = run_df.sort_values(step_col)
        first = run_df.iloc[0]
        final = run_df.iloc[-1]

        intervention_step = final.get("information_intervention_step", None)
        peer_intervention_step = final.get("peer_intervention_step", None)

        if pd.isna(intervention_step) and not pd.isna(peer_intervention_step):
            intervention_step = peer_intervention_step

        if pd.isna(intervention_step):
            pre_df = run_df[run_df[step_col] <= run_df[step_col].max() // 2]
            post_df = run_df[run_df[step_col] > run_df[step_col].max() // 2]
        else:
            intervention_step = int(intervention_step)
            pre_df = run_df[
                (run_df[step_col] >= max(0, intervention_step - 10))
                & (run_df[step_col] < intervention_step)
            ]
            post_df = run_df[
                (run_df[step_col] > intervention_step)
                & (run_df[step_col] <= intervention_step + 10)
            ]

        row = {
            **key_dict,
            "intervention_step": intervention_step,
            "initial_smoking_rate": first["Smoking rate"],
            "final_smoking_rate": final["Smoking rate"],
            "peak_smoking_rate": run_df["Smoking rate"].max(),
            "change_total": final["Smoking rate"] - first["Smoking rate"],
            "pre_smoking_rate": pre_df["Smoking rate"].mean(),
            "post_smoking_rate": post_df["Smoking rate"].mean(),
            "change_post_minus_pre": (
                post_df["Smoking rate"].mean()
                - pre_df["Smoking rate"].mean()
            )
        }

        for output in OUTPUTS:
            if output in final.index:
                row[f"final_{output}"] = final[output]
                row[f"mean_{output}"] = run_df[output].mean()

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("Scenarios to run:")
    for scenario_name in SCENARIOS:
        print("-", scenario_name)

    all_results = []

    for scenario_name, scenario_params in SCENARIOS.items():
        scenario_df = run_scenario(scenario_name, scenario_params)
        all_results.append(scenario_df)

    trajectories = pd.concat(all_results, ignore_index=True)
    run_summary = summarize_by_run(trajectories)

    scenario_summary = (
        run_summary
        .groupby("scenario")
        .mean(numeric_only=True)
        .reset_index()
    )

    trajectories.to_csv("batch_trajectories.csv", index=False)
    run_summary.to_csv("batch_run_summary.csv", index=False)
    scenario_summary.to_csv("batch_scenario_summary.csv", index=False)

    print("Saved batch_trajectories.csv")
    print("Saved batch_run_summary.csv")
    print("Saved batch_scenario_summary.csv")


if __name__ == "__main__":
    main()
