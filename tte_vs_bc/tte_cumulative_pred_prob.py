from toml import load as toml_load
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")

import numpy as np
from utils import get_df, collate
from embedding_model.metrics import kaplan_meier
from utils import format_model_config

from my_config import MyConfig
from embedding_model.config import Config

config = Config(**toml_load("config.toml"))
my_config = MyConfig(**toml_load("my_config.toml"))

LABELS = list(my_config.dict().keys())
CONFIGURATIONS = [("DTNN", 2020, 0), ("DCPH", 2020, 0)]

fig, axes = plt.subplots(1, len(LABELS), figsize=(16, 4))
for i, label in enumerate(LABELS):
    print(f"Processing {label}...")
    myconfig_phenotype = getattr(my_config, label)
    config_phenotype = getattr(config.phenotypes, label)
    num_bins = len(config_phenotype.bin_boundaries) - 1

    for j, configuration in enumerate(CONFIGURATIONS):
        model_type, yob_cutoff, followup_cutoff = configuration
        model_config = format_model_config(
            model_type, yob_cutoff, followup_cutoff, myconfig_phenotype.name
        )

        model = myconfig_phenotype.load_model(
            model_type, yob_cutoff, followup_cutoff, num_bins
        )

        df = get_df(
            myconfig_phenotype.filepaths.test1422,
            yob_cutoff,
            followup_cutoff,
            config_phenotype.age_cutoff,
        )

        test_s, test_t, test_predictions = collate(
            model, df, myconfig_phenotype.vocab, config.model.seq_threshold, label
        )

        if model_type == "DTNN":
            cumulative_predicted_risk = np.cumsum(test_predictions, axis=1)[:, :-1]

            cp_mean = [0] + list(cumulative_predicted_risk.mean(axis=0))
            cp_std = [0] + list(cumulative_predicted_risk.std(axis=0))
            times = config_phenotype.bin_boundaries[1:]

        elif model_type == "DCPH":
            baseline_hazards = model.compute_baseline_hazards(
                test_predictions, test_s, test_t
            )
            predicted_cumulative_hazards = model.predict_cumulative_hazards(
                baseline_hazards, test_predictions
            )
            predicted_cumulative_probs = 1 - np.exp(-predicted_cumulative_hazards)
            cp_mean = [0] + list(predicted_cumulative_probs.mean(axis=1))
            cp_std = [0] + list(predicted_cumulative_probs.std(axis=1))
            times = predicted_cumulative_probs.index.values

        ax = axes[i]
        ax.plot(
            [0] + list(times),
            cp_mean,
            label=model_config,
        )
        ax.fill_between(
            [0] + list(times),
            np.array(cp_mean) - np.array(cp_std),
            np.array(cp_mean) + np.array(cp_std),
            alpha=0.3,
        )

    km_times, km_mean, km_var = kaplan_meier(test_s, test_t)
    ax.plot(
        km_times,
        1 - km_mean,
        linestyle="--",
        label="Kaplan Meier Curve",
        color="gray",
    )
    ax.fill_between(
        km_times,
        1 - km_mean - np.sqrt(km_var),
        1 - km_mean + np.sqrt(km_var),
        color="gray",
        alpha=0.3,
    )

    ax.set_xlabel("Years")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(myconfig_phenotype.name)
    ax.legend()


plt.tight_layout()
plt.savefig(f"data/results/tte_cumulative_pred_prob.png")
