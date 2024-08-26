import numpy as np
import pandas as pd
import polars as pl
from scipy.special import expit
import seaborn as sns
from embedding_model.utils import get_inputs, get_logits
from embedding_model.metrics import xAUCt, xAPt
import textwrap
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def format_model_config(model, yob_cutoff, followup_cutoff, label=""):
    # mapping
    if label == "RECURRENT OM":
        label = "ROM"
    if label == "FOOD ALLERGY":
        label = "FA"

    if model == "DTNN_SYN":
        title = rf"$\mathrm{{DTNN_{{YOB\leq{yob_cutoff}}}^{{{label},ss}}}}$"
    elif model == "BC_SYN":
        title = rf"$\mathrm{{BC_{{YOB\leq{yob_cutoff}}}^{{{label},ss}}}}$"
    else:
        if followup_cutoff > 0:
            title = rf"$\mathrm{{{model}_{{t\geq{followup_cutoff}}}^{{{label}}}}}$"
        elif followup_cutoff == 0:
            title = rf"$\mathrm{{{model}_{{YOB\leq{yob_cutoff}}}^{{{label}}}}}$"

    return title


def get_yob_dfs(df):
    yob_dfs = {}
    for year, yob_df in df.groupby(df["date_of_birth"].dt.year):
        yob_dfs[year] = yob_df

    return yob_dfs


def get_followup_dfs(df):
    followup_dfs = {}
    for year, followup_df in df.groupby(df["censoring_age"].astype("int")):
        followup_dfs[year] = followup_df

    return followup_dfs


def get_event_count_group_dfs(df, label):
    df["age_less_than_diagnosis"] = (df["age"] < df[label + "_diagnosis_age"]).astype(
        int
    )
    patient_event_counts = (
        df.groupby("mrn")["age_less_than_diagnosis"].sum().astype(int)
    )
    df["event_count"] = patient_event_counts.loc[df["mrn"]].values

    # patient_event_counts = df.groupby("mrn").size()
    # df["event_count"] = patient_event_counts.loc[df["mrn"]].values
    sorted_patients = patient_event_counts.sort_values().index

    group_assignments = pd.qcut(
        patient_event_counts.loc[sorted_patients],
        q=5,
        labels=False,
        duplicates="drop",
    )

    df["group_num"] = df["mrn"].map(dict(zip(sorted_patients, group_assignments)))
    group_stats = df.groupby("group_num")["event_count"].agg(["min", "max"])
    group_stats["event_count_group"] = group_stats.apply(
        lambda row: f"{int(row['min'])}-{int(row['max'])}", axis=1
    )
    grouped_df = pd.merge(
        df,
        group_stats[["event_count_group"]],
        left_on="group_num",
        right_index=True,
        how="left",
    )

    sorted_keys = sorted(
        grouped_df.groupby("event_count_group").apply(
            lambda x: (int(x.name.split("-")[0]), int(x.name.split("-")[1]), x)
        )
    )

    event_count_group_dfs = {
        f"{start}-{end}": group_df for start, end, group_df in sorted_keys
    }

    return event_count_group_dfs


def get_sex_dfs(df):
    sex_dfs = {}
    for sex, sex_df in df.groupby(df["sex"]):
        sex_dfs[sex] = sex_df

    order = ["Male", "Female"]
    sex_dfs_sorted = {key: sex_dfs[key] for key in order if key in sex_dfs}

    return sex_dfs_sorted


def map_race(race):
    if race == "Caucasian/White":
        return "White or Caucasian"
    elif race in ["Black or African American", "Asian", "Unavailable"]:
        return race
    else:
        return "Others"


def get_race_dfs(df):
    df["mapped_race"] = df["race"].apply(map_race)
    race_dfs = {}

    for race, race_df in df.groupby(df["mapped_race"]):
        race_dfs[race] = race_df

    order = [
        "White or Caucasian",
        "Black or African American",
        "Asian",
        "Unavailable",
        "Others",
    ]
    race_dfs_sorted = {key: race_dfs[key] for key in order if key in race_dfs}

    return race_dfs_sorted


def map_insurance(insurance):
    public_insurance = [
        "Medicaid",
        "OOS Medicaid",
        "Medicaid Pending",
        "Other Government",
        "NC Medicaid Managed Care",
        "NC MEDICAID",
    ]
    private_insurance = [
        "Managed Care",
        "OOS Blue Cross",
        "Commercial",
        "Medicare Advantage",
        "NC Blue Cross",
    ]

    if insurance in public_insurance:
        return "Public"
    elif insurance in private_insurance:
        return "Private"
    else:
        return "Others"


def get_insurance_dfs(df):
    df["mapped_insurance"] = df["financial_class"].apply(map_insurance)
    insurance_dfs = {}

    for insurance, insurance_df in df.groupby(df["mapped_insurance"]):
        insurance_dfs[insurance] = insurance_df

    order = ["Public", "Private", "Others"]
    insurance_dfs_sorted = {
        key: insurance_dfs[key] for key in order if key in insurance_dfs
    }

    return insurance_dfs_sorted


def get_df(filepath, yob_cutoff, followup_cutoff, age_cutoff):
    df = pd.read_parquet(path=filepath)
    df = filter_df_by_yob(df, yob_cutoff)
    df = df[df["censoring_age"] >= followup_cutoff]
    df = df[df["censoring_age"] >= age_cutoff]
    df = df[df["age"] < age_cutoff]

    return df


def filter_df_by_yob(df, year):
    if year == 2018:
        # hardcoded based on last duhs encounter to include only those born before 2018-06-02
        df = df[df["date_of_birth"].dt.date <= pd.Timestamp(f"{year}-06-02").date()]
    else:
        df = df[df["date_of_birth"].dt.year <= year]

    return df


def exclude_df_by_dob(df, lower_limit, upper_limit):
    # dates are inclusive
    return df[
        ~(
            (df["date_of_birth"].dt.date >= pd.Timestamp(lower_limit).date())
            & (df["date_of_birth"].dt.date <= pd.Timestamp(upper_limit).date())
        )
    ]


def get_mrn_dfs(df):
    return [group for _, group in df.groupby("mrn", sort=False)]


def collate(model, df, vocab, seq_threshold, label):
    mrn_dfs = get_mrn_dfs(df)

    labels = []
    event_times = []
    predictions = []

    for mrn_df in mrn_dfs:
        target = mrn_df[label].iloc[0]
        labels.append(target)

        event_time = mrn_df[f"{label}_time_to_event"].iloc[0]
        event_times.append(event_time)

        events = get_inputs(mrn_df, label, vocab, seq_threshold)
        prediction = get_logits(model, events.unsqueeze(0))
        predictions.append(prediction.squeeze())

    return np.array(labels), np.array(event_times), np.array(predictions)


def collate_x(model, x_dfs, vocab, seq_threshold, label):
    x_labels = []
    x_event_times = []
    x_predictions = []

    for x, x_df in x_dfs.items():
        labels, event_times, predictions = collate(
            model, x_df, vocab, seq_threshold, label
        )

        x_labels.append(labels)
        x_event_times.append(event_times)
        x_predictions.append(predictions)

    return x_labels, x_event_times, x_predictions


def plot_x_distributions(
    axes, x_data, x_s, x_t, x_predictions, times, time_to_pred, color
):
    x_positions = np.arange(len(x_data))

    x_pred_risks = []
    for predictions in x_predictions:
        if predictions.ndim == 1:
            cumulative_predicted_risk = np.tile(
                expit(predictions[:, np.newaxis]), (1, len(times))
            )
        elif predictions.ndim == 2:
            cumulative_predicted_risk = np.cumsum(predictions, axis=1)[:, :-1]
        x_pred_risks.append(cumulative_predicted_risk)

    # Predicted probability at tmax
    probability_tmax = [pred_risk[:, -1] for pred_risk in x_pred_risks]
    medians = [np.median(pred_risk) for pred_risk in probability_tmax]
    print(medians)

    sns.boxplot(
        data=probability_tmax,
        color=color,
        ax=axes[0],
        showfliers=False,
        boxprops=dict(linewidth=0),
        saturation=1,
    )
    axes[0].set_ylabel("Predicted probability")

    # Number of cases
    label_sum = [sum(s) for s in x_s]
    axes[1].bar(x=x_positions, height=label_sum, color=color)
    axes[1].set_ylabel("Number of cases")

    # Prevalence
    n_total = [len(s) for s in x_s]
    prevalence = [cases / total for cases, total in zip(label_sum, n_total)]
    axes[2].bar(x=x_positions, height=prevalence, color=color)
    axes[2].set_ylabel("Prevalence")

    """
    # AUC and AP (with confidence intervals)
    aucs = []
    aps = []
    n_bootstrap = 100
    time_index = np.where(times == time_to_pred)[0][0]

    for s, t, pred_risk in zip(x_s, x_t, x_pred_risks):
        bootstrap_aucs = []
        bootstrap_aps = []
        for _ in range(n_bootstrap):
            indices = np.arange(len(pred_risk))
            bootstrap_indices = np.random.choice(indices, len(indices), replace=True)

            bootstrap_auc = xAUCt(
                s[bootstrap_indices],
                t[bootstrap_indices],
                pred_risk[bootstrap_indices],
                times,
            )
            bootstrap_ap = xAPt(
                s[bootstrap_indices],
                t[bootstrap_indices],
                pred_risk[bootstrap_indices],
                times,
            )

            bootstrap_aucs.append(bootstrap_auc[time_index])
            bootstrap_aps.append(bootstrap_ap[time_index])

        aucs.append(bootstrap_aucs)
        aps.append(bootstrap_aps)

    auc_means = np.array([np.mean(auc) for auc in aucs])
    auc_low = np.array([np.percentile(auc, 2.5) for auc in aucs])
    auc_high = np.array([np.percentile(auc, 97.5) for auc in aucs])

    axes[3].bar(
        x_positions,
        auc_means,
        yerr=np.abs([auc_means - auc_low, auc_high - auc_means]),
        capsize=5,
        color=color,
    )

    axes[3].set_ylabel(f"xAUC$_{{{time_to_pred}}}$")

    ap_means = np.array([np.mean(ap) for ap in aps])
    ap_low = np.array([np.percentile(ap, 2.5) for ap in aps])
    ap_high = np.array([np.percentile(ap, 97.5) for ap in aps])

    axes[4].bar(
        x_positions,
        ap_means,
        yerr=np.abs([ap_means - ap_low, ap_high - ap_means]),
        capsize=5,
        color=color,
    )
    axes[4].set_ylabel(f"xAP$_{{{time_to_pred}}}$")
    """

    for ax in axes:
        ax.set_xticks(x_positions)
        ax.set_xticklabels([textwrap.fill(str(data), width=12) for data in x_data])
        ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
