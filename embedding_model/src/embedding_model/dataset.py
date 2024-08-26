import torch
from torch.utils.data import Dataset
import polars as pl
from datetime import datetime

from gensim.models import KeyedVectors
from embedding_model.utils import pad_sequences


class EHR_Dataset(Dataset):
    def __init__(self, config, label):
        self.config = config
        self.phenotype = getattr(config.phenotypes, label)
        self.vocab = self.get_vocab(self.phenotype.filepaths.word2vec)
        self.dfs = []

    def describe(self, subset):
        total_num = len(self.dfs)
        num_pos = sum(df[self.phenotype.label][0] for df in self.dfs)

        msg = f"This {subset} dataset contains a total of {total_num} patients, "
        msg += f"of which {num_pos} have diagnosis of {self.phenotype.label}."

        print(msg)

    def __len__(self):
        """
        Length special method, returns the number of patients in dataset.
        """

        return len(self.dfs)

    def get_dfs(self, filepath):
        """
        List of dataframes per MRN, filtered accordingly
        Faster to manipulate using polars vs pandas
        """
        df = pl.read_parquet(filepath)

        df = self.filter_df_by_yob(df, self.config.yob_cutoff)
        df = df.filter(pl.col("censoring_age") >= self.config.followup_cutoff)
        df = df.filter(pl.col("censoring_age") >= self.phenotype.age_cutoff)
        df = df.filter(pl.col("age") < self.phenotype.age_cutoff)
        
        dfs = df.partition_by("mrn")
        if self.config.experiment.semi_synthetic:
            semi_synthetic_dfs = []
            for mrn_df in dfs:
                t = mrn_df[f"{self.phenotype.label}_time_to_event"][0]
                s = mrn_df[self.phenotype.label][0]

                c = mrn_df["censoring_age"][0]
                c_adm = self.config.experiment.c_ss_max / self.config.experiment.t_max * c       
                t_ss = min(t, c_adm)

                if c_adm <= t:
                    s = 0
                
                mrn_df = mrn_df.with_columns(pl.col(f"{self.phenotype.label}_time_to_event").apply(lambda _: t_ss))
                mrn_df = mrn_df.with_columns(pl.col(self.phenotype.label).apply(lambda _: s))
                mrn_df = mrn_df.filter(pl.col("age") <= t_ss)

                if len(mrn_df) > 0:
                    semi_synthetic_dfs.append(mrn_df)
            
            return semi_synthetic_dfs
             
        return dfs

    def filter_df_by_yob(self, df, year):
        if year == 2018:
            # hardcoded based on last duhs encounter
            df = df.filter(pl.col('date_of_birth').dt.date() <= datetime(year, 6, 2))
        else:
            df = df.filter(df['date_of_birth'].dt.year() <= year)

        return df

    def get_vocab(self, filepath):
        """
        Returns word2vec vocabulary.
        """
        w2v = KeyedVectors.load(str(filepath))
        vocab = w2v.key_to_index

        return vocab

    def __getitem__(self, index):
        """
        Getitem special method, expects an integer value index, between 0 and len(self) - 1.
        Returns the events and true label, both in torch tensor format in dataset.
        """
        df = self.dfs[index]

        age_to_first_diagnosis = df[self.phenotype.label + "_diagnosis_age"][0]
        df_filtered = df.filter(pl.col("age") < age_to_first_diagnosis)

        if len(df_filtered) == 0:
            events = torch.full((self.config.model.seq_threshold,), self.vocab["PAD"])
        else:
            events = torch.tensor([self.vocab.get(event, self.vocab["OOV"]) for event in df_filtered["event"].to_list()])
            events = pad_sequences(events, self.config.model.seq_threshold, self.vocab["PAD"])
           
        label = torch.tensor(df[self.phenotype.label].head(1)).float()
        event_time = torch.tensor(df[f"{self.phenotype.label}_time_to_event"].head(1)).float()
        return events, label, event_time


class EHR_Train_Dataset(EHR_Dataset):
    def __init__(self, config, label):
        super().__init__(config, label)
        self.dfs = super().get_dfs(self.phenotype.filepaths.train)


class EHR_Val_Dataset(EHR_Dataset):
    def __init__(self, config, label):
        super().__init__(config, label)
        self.dfs = super().get_dfs(self.phenotype.filepaths.validation)


class EHR_Test_Dataset(EHR_Dataset):
    def __init__(self, config, label):
        super().__init__(config, label)
        self.dfs = super().get_dfs(self.phenotype.filepaths.test)
