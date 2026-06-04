from dataclasses import dataclass


@dataclass
class UpliftConfig:
    base_estimator: str = "xgboost"
    n_estimators: int = 400
    max_depth: int = 8
    learning_rate: float = 0.05
    random_state: int = 42
