from pydantic import BaseModel
from pathlib import Path
import torch
from typing import Optional, List
from gensim.models import KeyedVectors
from embedding_model.model import BC3, DTNN1, DCPH1


class Filepaths(BaseModel):
    checkpoint_dtnn2020: Path
    checkpoint_dtnn2020_synthetic: Path = None
    checkpoint_dcph2020: Path
    checkpoint_bc2020: Path
    checkpoint_bc2020_synthetic: Path = None
    checkpoint_bc2018: Path
    checkpoint_bct5: Path
    word2vec1422: Path
    train1422: Path
    val1422: Path
    test1422: Path


class Phenotype(BaseModel):
    name: str
    color: str
    eval_times: List[float]
    calibrated_time: float
    filepaths: Filepaths
    w2v: KeyedVectors = None
    vocab: Optional[dict] = dict()

    def __init__(self, **data):
        super().__init__(**data)
        self.w2v = KeyedVectors.load(str(self.filepaths.word2vec1422))
        self.vocab = self.w2v.key_to_index

    class Config:
        arbitrary_types_allowed = True

    def load_model(self, model_type, yob_cutoff, followup_cutoff, num_bins=None):
        if model_type == "DTNN":
            model = DTNN1(torch.tensor(self.w2v.vectors), num_bins)

            if yob_cutoff == 2020 and followup_cutoff == 0:
                checkpoint = self.filepaths.checkpoint_dtnn2020
        elif model_type == "DTNN_SYN":
            model = DTNN1(torch.tensor(self.w2v.vectors), num_bins)

            if yob_cutoff == 2020 and followup_cutoff == 0:
                checkpoint = self.filepaths.checkpoint_dtnn2020_synthetic
        elif model_type == "DCPH":
            model = DCPH1(torch.tensor(self.w2v.vectors))

            if yob_cutoff == 2020 and followup_cutoff == 0:
                checkpoint = self.filepaths.checkpoint_dcph2020
        elif model_type == "BC":
            model = BC3(torch.tensor(self.w2v.vectors))

            if yob_cutoff == 2020 and followup_cutoff == 0:
                checkpoint = self.filepaths.checkpoint_bc2020
            if yob_cutoff == 2018 and followup_cutoff == 0:
                checkpoint = self.filepaths.checkpoint_bc2018
            if yob_cutoff == 2020 and followup_cutoff == 5:
                checkpoint = self.filepaths.checkpoint_bct5

        elif model_type == "BC_SYN":
            model = BC3(torch.tensor(self.w2v.vectors))

            if yob_cutoff == 2020 and followup_cutoff == 0:
                checkpoint = self.filepaths.checkpoint_bc2020_synthetic

        model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))
        return model


class Autism(Phenotype):
    pass


class ADHD(Phenotype):
    pass


class EarInfection(Phenotype):
    pass


class FoodAllergy(Phenotype):
    pass


class MyConfig(BaseModel):
    autism: Autism
    adhd: ADHD
    ear_infection: EarInfection
    food_allergy: FoodAllergy
