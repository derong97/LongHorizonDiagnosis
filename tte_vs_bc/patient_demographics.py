from toml import load as toml_load
import pandas as pd
from scipy.stats import chi2_contingency
from utils import map_insurance, map_race
from embedding_model.config import Config

DATASET = "data/dataset/analytic(14-22).parquet"
LABELS = ["autism", "adhd", "ear_infection", "food_allergy"]
DEMOGRAPHICS_VAR = ["sex", "mapped_race", "mapped_insurance"]

config = Config(**toml_load("config.toml"))

df = pd.read_parquet(DATASET)
df = df[df["date_of_birth"].dt.year <= 2020]
df["mapped_race"] = df["race"].apply(map_race)
df["mapped_insurance"] = df["financial_class"].apply(map_insurance)

df = df.drop_duplicates(subset=["mrn"])

for demographics in DEMOGRAPHICS_VAR:
    value_counts = df[demographics].value_counts()
    print(f"Value counts: {value_counts}")

for label in LABELS:
    print(f"\n***{label}***")
    config_phenotype = getattr(config.phenotypes, label)
    print(f"original: {len(df)}")
    label_df = df[df[label + "_phenotype_age"] >= config_phenotype.age_cutoff]
    print(f"after removing positive indivs with phenotype age < age cutoff: {len(label_df)}")
    label_df = label_df[label_df["censoring_age"] >= config_phenotype.age_cutoff]
    print(f"after removing neg indivs with censoring age < age cutoff: {len(label_df)}")

    for demographics in DEMOGRAPHICS_VAR:
        print(f"\n==={demographics}===")

        contingency_table = pd.crosstab(label_df[demographics], label_df[label])
        chi2, p, _, _ = chi2_contingency(contingency_table)

        label_0_count = label_df[label].eq(0).sum()
        label_1_count = label_df[label].eq(1).sum()
        percentage_table = contingency_table.copy()
        percentage_table.iloc[:, 0] /= label_0_count
        percentage_table.iloc[:, 1] /= label_1_count
        percentage_table *= 100

        print(f"Label count: {label_1_count}")
        print("Contingency Table:")
        print(contingency_table)
        print("\nPercentage Table:")
        print(percentage_table)
        print(f"\nChi-square value: {chi2}")
        print(f"P-value: {p}")
