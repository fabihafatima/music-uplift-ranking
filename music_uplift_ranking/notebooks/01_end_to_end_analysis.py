"""
Notebook-style analysis script.
Run: python music_uplift_ranking/notebooks/01_end_to_end_analysis.py
Or paste cells into Jupyter after pipeline completes.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    campaigns = pd.read_parquet(ROOT / "data/raw/campaigns.parquet")
    ite = pd.read_parquet(ROOT / "artifacts/uplift/ite_predictions.parquet")
    top = pd.read_parquet(ROOT / "artifacts/top_promotion_candidates.parquet")

    print("Campaign summary")
    print(campaigns[["promoted", "streams", "true_ite_streams"]].describe())
    print("\nPromotion rate:", campaigns["promoted"].mean())
    print("\nTop 10 promotion candidates:")
    print(
        top.head(10)[
            ["track_id", "promotion_rank", "predicted_lift_score", "ranking_score", "genre"]
        ]
    )

    import json

    report = json.loads((ROOT / "artifacts/evaluation/evaluation_report.json").read_text())
    print("\nUplift Qini:", report["uplift"]["qini_coefficient"])
    print("Ranking nDCG@10:", report["ranking"]["ranking_model"]["ndcg@10"])
    print("vs historical baseline:", report["ranking"]["historical_streams"]["ndcg@10"])


if __name__ == "__main__":
    main()
