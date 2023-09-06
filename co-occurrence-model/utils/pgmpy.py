import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

__all__ = ["log_posterior"]


def log_posterior(df: pd.DataFrame, model: BayesianNetwork) -> float:
    score = 0

    engine = VariableElimination(model)
    for i in range(len(df)):
        data = df.iloc[i].to_dict()

        for obj in data:
            evi = {k: v for k, v in data.items() if k != obj}
            query = engine.query([obj], evi)
            score += np.log(query.values[data[obj]])

    return score
