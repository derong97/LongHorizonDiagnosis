from toml import load as toml_load
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")

import seaborn as sns
import pandas as pd
import numpy as np
from scipy.special import expit
from utils import get_df, exclude_df_by_dob, collate_x, format_model_config, get_yob_dfs
from my_config import MyConfig
from embedding_model.config import Config


config = Config(**toml_load("config.toml"))
my_config = MyConfig(**toml_load("my_config.toml"))

LABELS = list(my_config.dict().keys())
CONFIGURATIONS = [
    ("DTNN", 2020, 0),
    ("DCPH", 2020, 0),
    ("BC", 2020, 0),
    ("BC", 2018, 0),
    ("BC", 2020, 5),
]
years = list(range(2014, 2020 + 1))

fig, axes = plt.subplots(2, 2, figsize=(20, 12))
axes = axes.flatten()

for i, label in enumerate(LABELS):
    print(f"Processing {label}...")

    myconfig_phenotype = getattr(my_config, label)
    config_phenotype = getattr(config.phenotypes, label)

    bin_boundaries = config_phenotype.bin_boundaries

    df = get_df(
        myconfig_phenotype.filepaths.test1422,
        2020,
        0,
        config_phenotype.age_cutoff,
    )
    df_size = len(CONFIGURATIONS) * len(df)
    df_boxplot = pd.DataFrame(
        index=range(df_size), columns=["Years", "Model Set-up", "Predicted Probability"]
    )

    idx = 0
    for j, configuration in enumerate(CONFIGURATIONS):
        print(configuration)

        model_type, yob_cutoff, followup_cutoff = configuration
        model = myconfig_phenotype.load_model(
            model_type, yob_cutoff, followup_cutoff, len(bin_boundaries) - 1
        )
        model_config = format_model_config(
            model_type, yob_cutoff, followup_cutoff, myconfig_phenotype.name
        )

        if followup_cutoff == 5:
            df_2014_2018 = df[df["censoring_age"] > followup_cutoff]  # ID
            df_2019_2020 = df[df["date_of_birth"].dt.year >= 2019]  # OOD
            df_final = pd.concat([df_2014_2018, df_2019_2020], ignore_index=True)
        if yob_cutoff == 2018:
            df_final = exclude_df_by_dob(
                df, "2018-06-02", "2018-12-31"
            )  # make 2018 ID year

        else:
            df_final = df.copy()

        dfs = get_yob_dfs(df_final)

        times = bin_boundaries[1:]
        x_labels, x_event_times, x_predictions = collate_x(
            model, dfs, myconfig_phenotype.vocab, config.model.seq_threshold, label
        )

        years = list(dfs.keys())

        # Get predicted probability
        x_pred_risks = []
        for labels, event_times, predictions in zip(
            x_labels, x_event_times, x_predictions
        ):
            if model_type == "BC":
                assert predictions.ndim == 1
                cumulative_predicted_risk = np.tile(
                    expit(predictions[:, np.newaxis]), (1, len(times))
                )
            elif model_type == "DTNN":
                assert predictions.ndim == 2
                cumulative_predicted_risk = np.cumsum(predictions, axis=1)[:, :-1]
            elif model_type == "DCPH":
                assert predictions.ndim == 1
                baseline_hazards = model.compute_baseline_hazards(
                    predictions, labels, event_times
                )
                predicted_cumulative_hazards = model.predict_cumulative_hazards(
                    baseline_hazards, predictions
                )
                predicted_cumulative_probs = 1 - np.exp(-predicted_cumulative_hazards)
                cumulative_predicted_risk = predicted_cumulative_probs.values.T

            x_pred_risks.append(cumulative_predicted_risk)
        probability_tmax = [pred_risk[:, -1] for pred_risk in x_pred_risks]

        for year, year_probs in zip(years, probability_tmax):
            for indiv_prob in year_probs:
                df_boxplot.loc[idx] = [year, model_config, indiv_prob]
                idx += 1

    sns.boxplot(
        data=df_boxplot,
        x="Years",
        y="Predicted Probability",
        hue="Model Set-up",
        showfliers=False,
        boxprops=dict(linewidth=0),
        saturation=1,
        ax=axes[i],
    )

    axes[i].set_title(myconfig_phenotype.name, fontsize=15)
    handles, labels = axes[i].get_legend_handles_labels()
    axes[i].legend(handles=handles, labels=labels)

plt.tight_layout()
plt.savefig("data/results/subgroup_yob_grouped_probability_v1.png")
