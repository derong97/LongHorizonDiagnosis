from toml import load as toml_load
from utils import get_df, collate
from sksurv.metrics import concordance_index_censored
import numpy as np

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

for label in LABELS:
    print(f"Processing {label}...")
    myconfig_phenotype = getattr(my_config, label)
    config_phenotype = getattr(config.phenotypes, label)
    num_bins = len(config_phenotype.bin_boundaries) - 1

    for configuration in CONFIGURATIONS:
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

        if test_predictions.ndim == 2:
            test_predictions = np.cumsum(test_predictions, axis=1)[:, :-1][:, -1]

        concordance_index = concordance_index_censored(
            test_s.astype(bool), test_t, test_predictions
        )
        print(f"{configuration}: {concordance_index}")
