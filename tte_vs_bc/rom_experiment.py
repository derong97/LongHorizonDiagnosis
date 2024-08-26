from toml import load as toml_load
import numpy as np
from utils import (
    get_df,
    collate,
)
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")

from sklearn.calibration import calibration_curve
from embedding_model.metrics import one_calibration

from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score
from embedding_model.metrics import xAUCt, xAPt
from utils import format_model_config

from my_config import MyConfig
from embedding_model.config import Config


def map_color(model_type):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if model_type == "DTNN":
        return colors[0]
    elif model_type == "DTNN_SYN":
        return colors[1]
    elif model_type == "BC":
        return colors[2]
    elif model_type == "BC_SYN":
        return colors[3]


LABEL = "ear_infection"
CONFIGURATIONS = [
    ("DTNN", 2020, 0),
    ("DTNN_SYN", 2020, 0),
    ("BC", 2020, 0),
    ("BC_SYN", 2020, 0),
]
N_BINS = 10
N_BOOTSTRAP_SAMPLES = 100
TIME_TO_CALIBRATE = 1.0

config = Config(**toml_load("config.toml"))
my_config = MyConfig(**toml_load("my_config.toml"))
myconfig_phenotype = getattr(my_config, LABEL)
config_phenotype = getattr(config.phenotypes, LABEL)
bin_boundaries = config_phenotype.bin_boundaries
times = bin_boundaries[1:]
time_indices = np.searchsorted(times, myconfig_phenotype.eval_times)
calibrated_time_index = np.where(times == TIME_TO_CALIBRATE)[0][0]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

df = get_df(
    myconfig_phenotype.filepaths.test1422,
    2020,
    0,
    config_phenotype.age_cutoff,
)

aucs = []
aps = []
model_types = []
max_val = 0
for index, configuration in enumerate(CONFIGURATIONS):
    print(configuration)

    model_type, yob_cutoff, followup_cutoff = configuration
    model = myconfig_phenotype.load_model(
        model_type, yob_cutoff, followup_cutoff, len(bin_boundaries) - 1
    )
    model_config = format_model_config(
        model_type, yob_cutoff, followup_cutoff, myconfig_phenotype.name
    )

    test_s, test_t, test_predictions = collate(
        model, df, myconfig_phenotype.vocab, config.model.seq_threshold, LABEL
    )
    if test_predictions.ndim == 1:
        cumulative_predicted_risk = np.tile(
            expit(test_predictions[:, np.newaxis]), (1, len(times))
        )
    elif test_predictions.ndim == 2:
        cumulative_predicted_risk = np.cumsum(test_predictions, axis=1)[:, :-1]

    probability_tmax = cumulative_predicted_risk[:, -1]
    auct, auct_low, auct_high = xAUCt(
        test_s,
        test_t,
        cumulative_predicted_risk,
        times,
        n_bootstrap_samples=N_BOOTSTRAP_SAMPLES,
    )
    (apt, prevt), (apt_low, prev_low), (apt_high, prev_high) = xAPt(
        test_s,
        test_t,
        cumulative_predicted_risk,
        times,
        return_prevalence=True,
        n_bootstrap_samples=N_BOOTSTRAP_SAMPLES,
    )
    auc = roc_auc_score(test_s, probability_tmax)
    ap = average_precision_score(test_s, probability_tmax)

    print(f"AUCt: {auct}")
    print(f"APt: {apt}")
    print(f"AUC: {auc}")
    print(f"AP: {ap}")

    bootstrap_aucs = []
    bootstrap_aps = []
    for _ in range(N_BOOTSTRAP_SAMPLES):
        indices = np.arange(len(test_s))
        bootstrap_indices = np.random.choice(indices, len(indices), replace=True)

        bootstrap_auc = roc_auc_score(
            test_s[bootstrap_indices], probability_tmax[bootstrap_indices]
        )
        bootstrap_ap = average_precision_score(
            test_s[bootstrap_indices], probability_tmax[bootstrap_indices]
        )
        bootstrap_aucs.append(bootstrap_auc)
        bootstrap_aps.append(bootstrap_ap)

    aucs.append(bootstrap_aucs)
    aps.append(bootstrap_aps)
    model_types.append(model_type)

    # Calibration curves
    if model_type == "DTNN" or model_type == "DTNN_SYN":
        _, _, _, op, ep = one_calibration(
            test_s, test_t, test_predictions, times, return_curves=True
        )

        prob_pred = op[calibrated_time_index]
        prob_true = ep[calibrated_time_index]

        calibration_time_text = f" (t={TIME_TO_CALIBRATE})"

    elif model_type == "BC" or model_type == "BC_SYN":
        prob_true, prob_pred = calibration_curve(
            test_s, expit(test_predictions), n_bins=N_BINS, strategy="quantile"
        )

        calibration_time_text = ""

    if max(prob_true) > max_val:
        max_val = max(prob_true)
    if max(prob_pred) > max_val:
        max_val = max(prob_pred)

    axes[0].plot(
        prob_pred,
        prob_true,
        marker="o",
        label=model_config + calibration_time_text,
        color=map_color(model_type),
    )
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives")

    # AUCt
    axes[1].plot(
        myconfig_phenotype.eval_times,
        auct[time_indices],
        label=model_config,
        color=map_color(model_type),
    )
    axes[1].fill_between(
        myconfig_phenotype.eval_times,
        auct_low[time_indices],
        auct_high[time_indices],
        color=map_color(model_type),
        alpha=0.1,
    )

    axes[1].set_ylabel("AUC$_t$")
    axes[1].set_xlabel("Years (t)")

# Regular AUC
auc_means = np.array([np.mean(auc) for auc in aucs])
auc_low = np.array([np.percentile(auc, 2.5) for auc in aucs])
auc_high = np.array([np.percentile(auc, 97.5) for auc in aucs])

axes[2].bar(
    model_types,
    auc_means,
    yerr=[auc_means - auc_low, auc_high - auc_means],
    capsize=5,
    color=[map_color(model_type) for model_type in model_types],
)
axes[2].set_xticks(range(len(model_types)))
axes[2].set_xticklabels(
    [
        format_model_config(*configuration, myconfig_phenotype.name)
        for configuration in CONFIGURATIONS
    ],
    rotation=45,
)
axes[2].set_ylabel("Regular AUC")


# Reference points
axes[0].plot(
    [0, max_val],
    [0, max_val],
    linestyle="--",
    label="Perfectly Calibrated",
    color="gray",
)

axes[1].plot(
    myconfig_phenotype.eval_times,
    np.array(myconfig_phenotype.eval_times) * 0 + 0.5,
    "--",
    color="gray",
)
axes[2].axhline(y=0.5, linestyle="--", color="gray")

axes[0].legend()
axes[1].legend()

plt.tight_layout()
plt.savefig("data/results/rom_experiment.png")
