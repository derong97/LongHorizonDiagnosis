from toml import load as toml_load
from utils import get_df, collate
from sksurv.metrics import concordance_index_censored

from my_config import MyConfig
from embedding_model.config import Config

config = Config(**toml_load("config.toml"))
my_config = MyConfig(**toml_load("my_config.toml"))

LABELS = list(my_config.dict().keys())
CONFIGURATIONS = [("BC", 2020, 0), ("BC", 2018, 0), ("BC", 2020, 5)]

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
        censored_times = (
            df[["mrn", f"censoring_age"]].groupby("mrn", sort=False).first()
        ).squeeze()

        censoring_indicator = 1 - test_s
        concordance_index = concordance_index_censored(
            censoring_indicator.astype(bool), censored_times, 1 - test_predictions
        )
        print(f"{configuration}: {concordance_index}")
