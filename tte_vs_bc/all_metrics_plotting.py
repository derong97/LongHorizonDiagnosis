from toml import load as toml_load
import numpy as np
from utils import (
    get_df,
    collate,
)
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("muted")

from sklearn.metrics import roc_auc_score, average_precision_score
from embedding_model.metrics import xAUCt, xAPt
from utils import format_model_config

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

N_BOOTSTRAP_SAMPLES = 100

fig_auc, axes_auc = plt.subplots(2, len(LABELS), figsize=(16, 8))
fig_ap, axes_ap = plt.subplots(2, len(LABELS), figsize=(16, 8))
for index, label in enumerate(LABELS):
    print(f"Processing {label}...")

    myconfig_phenotype = getattr(my_config, label)
    config_phenotype = getattr(config.phenotypes, label)

    bin_boundaries = config_phenotype.bin_boundaries
    times = bin_boundaries[1:]
    time_indices = np.searchsorted(times, myconfig_phenotype.eval_times)

    x_data = []
    aucs = []
    aps = []
    prevs = []

    for j, configuration in enumerate(CONFIGURATIONS):
        print(configuration)

        model_type, yob_cutoff, followup_cutoff = configuration
        model = myconfig_phenotype.load_model(
            model_type, yob_cutoff, followup_cutoff, len(bin_boundaries) - 1
        )
        model_config = format_model_config(
            model_type, yob_cutoff, followup_cutoff, myconfig_phenotype.name
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

        if test_predictions.ndim == 1:
            cumulative_predicted_risk = np.tile(
                test_predictions[:, np.newaxis], (1, len(times))
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
        prev = sum(test_s) / len(test_s)

        print(f"auct: {np.round(auct, 3)}")
        print(f"auct low: {auct_low}")
        print(f"auct high: {auct_high}")
        print(f"apt: {np.round(apt, 3)}")
        print(f"apt low: {apt_low}")
        print(f"apt high: {apt_high}")

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
        prevs.append(prev)
        x_data.append(model_config)

        # auct
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][j]

        axes_auc[0, index].plot(
            myconfig_phenotype.eval_times,
            auct[time_indices],
            label=model_config,
            color=color,
        )
        axes_auc[0, index].fill_between(
            myconfig_phenotype.eval_times,
            auct_low[time_indices],
            auct_high[time_indices],
            color=color,
            alpha=0.2,
        )
        axes_auc[0, index].plot(
            myconfig_phenotype.eval_times,
            np.array(myconfig_phenotype.eval_times) * 0 + 0.5,
            "--",
            color="gray",
            alpha=0.5,
        )
        axes_auc[0, index].set_ylabel("AUC$_t$")
        axes_auc[0, index].set_xlabel("Years (t)")
        axes_auc[0, index].legend()

        # APt
        axes_ap[0, index].plot(
            myconfig_phenotype.eval_times,
            apt[time_indices],
            label=model_config,
            color=color,
        )
        axes_ap[0, index].fill_between(
            myconfig_phenotype.eval_times,
            apt_low[time_indices],
            apt_high[time_indices],
            color=color,
            alpha=0.2,
        )
        axes_ap[0, index].plot(
            myconfig_phenotype.eval_times,
            prevt[time_indices],
            "--",
            color=color,
            alpha=0.5,
        )
        axes_ap[0, index].set_ylabel("AP$_t$")
        axes_ap[0, index].set_xlabel("Years (t)")
        axes_ap[0, index].legend()

    # regular auc
    auc_means = np.array([np.mean(auc) for auc in aucs])
    auc_low = np.array([np.percentile(auc, 2.5) for auc in aucs])
    auc_high = np.array([np.percentile(auc, 97.5) for auc in aucs])

    print(f"auc: {auc_means})")
    print(f"auc low: {auc_low}")
    print(f"auc high: {auc_high}")

    axes_auc[1, index].bar(
        x_data,
        auc_means,
        yerr=[auc_means - auc_low, auc_high - auc_means],
        capsize=5,
        color=myconfig_phenotype.color,
    )
    axes_auc[1, index].set_xticks(range(len(x_data)))
    axes_auc[1, index].set_xticklabels(x_data, rotation=45)
    axes_auc[1, index].axhline(y=0.5, linestyle="--", color="gray", alpha=0.5)
    axes_auc[1, index].set_ylabel("Regular AUC")

    # regular ap
    ap_means = np.array([np.mean(ap) for ap in aps])
    ap_low = np.array([np.percentile(ap, 2.5) for ap in aps])
    ap_high = np.array([np.percentile(ap, 97.5) for ap in aps])

    print(f"ap: {ap_means})")
    print(f"ap low: {ap_low}")
    print(f"ap high: {ap_high}")

    axes_ap[1, index].bar(
        x_data,
        ap_means,
        yerr=[ap_means - ap_low, ap_high - ap_means],
        capsize=5,
        color=myconfig_phenotype.color,
    )
    for x, prev in enumerate(prevs):
        axes_ap[1, index].plot(
            [x - 0.45, x + 0.45], [prev, prev], ls="--", c="k", alpha=0.5
        )
    axes_ap[1, index].set_xticks(range(len(x_data)))
    axes_ap[1, index].set_xticklabels(x_data, rotation=45)
    axes_ap[1, index].set_ylabel("Regular AP")

    # Set title
    axes_auc[0, index].set_title(myconfig_phenotype.name, fontsize=15)
    axes_ap[0, index].set_title(myconfig_phenotype.name, fontsize=15)

fig_auc.tight_layout()
fig_ap.tight_layout()
fig_auc.savefig(f"data/results/all_metrics_auc.png")
fig_ap.savefig(f"data/results/all_metrics_ap.png")
