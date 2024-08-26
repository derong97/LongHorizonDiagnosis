from pydantic import BaseModel, validator
from pathlib import Path
from typing import List
import numpy as np

class Filepaths(BaseModel):
    train: Path
    validation: Path
    test: Path
    word2vec: Path

class Phenotype(BaseModel):
    label: str
    age_cutoff: float
    bin_boundaries: List[float]
    filepaths: Filepaths

    @validator("bin_boundaries", pre=False)
    def convert_to_numpy_array(cls, v):
        return np.array(v)

class Autism(Phenotype):
    pass

class ADHD(Phenotype):
    pass

class EarInfection(Phenotype):
    pass

class FoodAllergy(Phenotype):
    pass


class Phenotypes(BaseModel):
    autism: Autism
    adhd: ADHD
    ear_infection: EarInfection
    food_allergy: FoodAllergy


class Model(BaseModel):
    seq_threshold: int
    num_layers: int
    num_heads: int
    dropout: float


class Training(BaseModel):
    max_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int


class Experiment(BaseModel):
    semi_synthetic: bool
    c_ss_max: float
    t_max: float
    

class Config(BaseModel):
    yob_cutoff: int
    followup_cutoff: int
    phenotypes: Phenotypes
    model: Model
    training: Training
    experiment: Experiment
