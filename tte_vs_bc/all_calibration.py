from toml import load as toml_load
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")

import numpy as np
from scipy.special import expit
from utils import get_df, collate, format_model_config
from embedding_model.metrics import one_calibration

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
N_BINS = 10

fig, axes = plt.subplots(len(CONFIGURATIONS), len(LABELS), figsize=(20, 20))
for i, label in enumerate(LABELS):
    print(f"Processing {label}...")
    myconfig_phenotype = getattr(my_config, label)
    config_phenotype = getattr(config.phenotypes, label)
    num_bins = len(config_phenotype.bin_boundaries) - 1
    times = config_phenotype.bin_boundaries[1:]
    time_indices = np.searchsorted(times, myconfig_phenotype.eval_times)
    calibrated_time_index = np.where(times == myconfig_phenotype.calibrated_time)[0][0]

    for j, configuration in enumerate(CONFIGURATIONS):
        print(configuration)
        ax = axes[j, i]

        model_type, yob_cutoff, followup_cutoff = configuration
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

        if model_type == "DTNN" or model_type == "DCPH":
            if model_type == "DCPH":
                baseline_hazards = model.compute_baseline_hazards(
                    test_predictions, test_s, test_t
                )
                predicted_cumulative_hazards = model.predict_cumulative_hazards(
                    baseline_hazards, test_predictions
                )
                predicted_cumulative_probs = 1 - np.exp(-predicted_cumulative_hazards)
                binned_cumulative_predictions = np.array(
                    [
                        np.interp(
                            times,
                            predicted_cumulative_probs.index,
                            predicted_cumulative_probs[col],
                        )
                        for col in predicted_cumulative_probs.columns
                    ]
                )
                # calculate probability for each bin to use the one calibration function
                test_predictions = np.diff(
                    binned_cumulative_predictions, axis=1, prepend=0
                )

            _, _, _, op, ep = one_calibration(
                test_s, test_t, test_predictions, times, return_curves=True
            )

            prob_pred = op[calibrated_time_index]
            prob_true = ep[calibrated_time_index]

            calibration_time_text = f" (t={myconfig_phenotype.calibrated_time})"

            max_val = 0
            if max(prob_pred) > max_val:
                max_val = max(prob_pred)
            if max(prob_true) > max_val:
                max_val = max(prob_true)

            ax.plot(
                prob_pred,
                prob_true,
                marker="o",
                label="Calibrated ID Curve" + calibration_time_text,
                color=myconfig_phenotype.color,
            )

        elif model_type == "BC":
            prob_true, prob_pred = calibration_curve(
                test_s, expit(test_predictions), n_bins=N_BINS, strategy="quantile"
            )
            max_val = 0
            if max(prob_true) > max_val:
                max_val = max(prob_true)
            if max(prob_pred) > max_val:
                max_val = max(prob_pred)

            ax.plot(
                prob_pred,
                prob_true,
                marker="o",
                label=f"Calibrated ID Curve",
                color=myconfig_phenotype.color,
            )

            # Plot OOD calibration curves
            if yob_cutoff == 2018 or followup_cutoff == 5:
                df = get_df(
                    myconfig_phenotype.filepaths.test1422,
                    2020,
                    0,
                    config_phenotype.age_cutoff,
                )
                df_2019_2020 = df[df["date_of_birth"].dt.year >= 2019]  # OOD

                test_s, test_t, test_predictions = collate(
                    model,
                    df_2019_2020,
                    myconfig_phenotype.vocab,
                    config.model.seq_threshold,
                    label,
                )
                prob_true, prob_pred = calibration_curve(
                    test_s, expit(test_predictions), n_bins=N_BINS, strategy="quantile"
                )
                if max(prob_true) > max_val:
                    max_val = max(prob_true)
                if max(prob_pred) > max_val:
                    max_val = max(prob_pred)

                ax.plot(
                    prob_pred,
                    prob_true,
                    marker="x",
                    label=f"Calibrated OOD Curve",
                    color=myconfig_phenotype.color,
                )

        ax.plot(
            [0, max_val],
            [0, max_val],
            linestyle="--",
            label="Perfectly Calibrated",
            color="gray",
        )
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend()


for i, label in enumerate(LABELS):
    axes[0, i].spines["top"].set_visible(False)
    myconfig_phenotype = getattr(my_config, label)
    axes[0, i].text(
        0.5,
        1.05,
        myconfig_phenotype.name,
        rotation=0,
        ha="center",
        va="center",
        transform=axes[0, i].transAxes,
        fontsize=20,
    )

for j, (model_type, yob_cutoff, followup_cutoff) in enumerate(CONFIGURATIONS):
    axes[j, 0].spines["left"].set_visible(False)
    axes[j, 0].text(
        -0.2,
        0.5,
        format_model_config(model_type, yob_cutoff, followup_cutoff),
        rotation=90,
        ha="center",
        va="center",
        transform=axes[j, 0].transAxes,
        fontsize=20,
    )


plt.tight_layout(rect=[0.02, 0, 1, 1])
plt.savefig(f"data/results/all_calibration_curves.png")
