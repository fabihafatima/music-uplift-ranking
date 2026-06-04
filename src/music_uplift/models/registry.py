"""Model factory registry."""

from __future__ import annotations

from music_uplift.config import ModelConfig
from music_uplift.models.base import UpliftModel
from music_uplift.models.r_learner import RLearner
from music_uplift.models.t_learner import TLearner
from music_uplift.models.x_learner import XLearner

LEARNERS = {
    "t_learner": TLearner,
    "x_learner": XLearner,
    "r_learner": RLearner,
}


def create_uplift_model(config: ModelConfig) -> UpliftModel:
    cls = LEARNERS.get(config.learner)
    if cls is None:
        raise ValueError(f"Unknown learner '{config.learner}'. Choose from: {list(LEARNERS)}")
    return cls(config)
