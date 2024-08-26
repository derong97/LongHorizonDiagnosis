import argparse
import polars as pl
import pandas as pd
from embedding_model.utils import train_w2v
from sklearn.model_selection import train_test_split


def train_validation_test_split(df: pl.DataFrame, train_size: float, label: str):
    dfs = df.partition_by("mrn")
    labels = [mrn_df[label][0] for mrn_df in dfs]

    train, test_validation = train_test_split(
        dfs, train_size=train_size, stratify=labels, shuffle=True
    )

    labels = [mrn_df[label][0] for mrn_df in test_validation]
    validation, test = train_test_split(
        test_validation, train_size=0.5, stratify=labels, shuffle=True
    )
    return pl.concat(train), pl.concat(validation), pl.concat(test)


parser = argparse.ArgumentParser(description="Train-val-test in-distribution splits")
parser.add_argument(
    "--label",
    type=str,
    required=True,
    choices=["autism", "adhd", "ear_infection", "food_allergy"],
)
parser.add_argument("--train_size", type=float, default=0.6)

args = parser.parse_args()

analytic_location = "data/dataset/analytic(14-22).parquet"
train_location = f"data/dataset/{args.label}/train(14-22).parquet"
val_location = f"data/dataset/{args.label}/val(14-22).parquet"
test_location = f"data/dataset/{args.label}/test(14-22).parquet"
word2vec_location = f"data/dataset/{args.label}/w2v(14-22).kvmodel"

df = pl.read_parquet(analytic_location)
train, validation, test = train_validation_test_split(df, args.train_size, args.label)
train.write_parquet(train_location)
validation.write_parquet(val_location)
test.write_parquet(test_location)

df_train = pd.read_parquet(train_location)
sentences = df_train.groupby("mrn")["event"].agg(list).to_list()
w2v_model = train_w2v(sentences)
w2v_model.wv.save(word2vec_location)
