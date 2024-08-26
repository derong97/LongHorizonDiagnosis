from toml import load as toml_load
import numpy as np
from utils import (
    get_df,
    collate,
)
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")

from embedding_model.metrics import kaplan_meier
from utils import format_model_config

from my_config import MyConfig
from embedding_model.config import Config

LABEL = "ear_infection"
CONFIGURATIONS = [
    ("DTNN", 2020, 0),
    ("DTNN_SYN", 2020, 0),
]

config = Config(**toml_load("config.toml"))
my_config = MyConfig(**toml_load("my_config.toml"))
myconfig_phenotype = getattr(my_config, LABEL)
config_phenotype = getattr(config.phenotypes, LABEL)

df = get_df(
    myconfig_phenotype.filepaths.test1422,
    2020,
    0,
    config_phenotype.age_cutoff,
)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
for index, configuration in enumerate(CONFIGURATIONS):
    print(configuration)

    model_type, yob_cutoff, followup_cutoff = configuration
    model = myconfig_phenotype.load_model(
        model_type,
        yob_cutoff,
        followup_cutoff,
        len(config_phenotype.bin_boundaries) - 1,
    )
    model_config = format_model_config(
        model_type, yob_cutoff, followup_cutoff, myconfig_phenotype.name
    )

    test_s, test_t, test_predictions = collate(
        model, df, myconfig_phenotype.vocab, config.model.seq_threshold, LABEL
    )

    times = config_phenotype.bin_boundaries[1:]
    cumulative_predicted_risk = np.cumsum(test_predictions, axis=1)[:, :-1]

    cp_mean = [0] + list(cumulative_predicted_risk.mean(axis=0))
    cp_std = [0] + list(cumulative_predicted_risk.std(axis=0))

    km_times, km_mean, km_var = kaplan_meier(test_s, test_t)

    ax = axes[index]
    ax.plot(
        [0] + list(times),
        cp_mean,
        marker="o",
        label=f"Cumulative Probability Curve",
        color=myconfig_phenotype.color,
    )
    ax.fill_between(
        [0] + list(times),
        np.array(cp_mean) - np.array(cp_std),
        np.array(cp_mean) + np.array(cp_std),
        color=myconfig_phenotype.color,
        alpha=0.3,
    )
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
    ax.set_ylabel("Failure Probability")

    ax.set_title(myconfig_phenotype.name)
    ax.legend()

plt.tight_layout()
plt.savefig(f"data/results/dtnn_rom.png")
