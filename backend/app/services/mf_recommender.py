from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD


class MFRecommender:
    def __init__(self, n_factors: int = 20):
        self.n_factors = int(max(2, n_factors))
        self.svd: TruncatedSVD | None = None
        self.fitted = False
        self.item_ids: list[str] = []
        self.user_ids: list[str] = []
        self.U: np.ndarray | None = None
        self.V: np.ndarray | None = None

    def fit(self, wear_matrix: np.ndarray, item_ids: Sequence[str], user_ids: Sequence[str]) -> "MFRecommender":
        matrix = np.asarray(wear_matrix, dtype=np.float32)
        if matrix.ndim != 2:
            self.fitted = False
            return self
        n_users, n_items = matrix.shape
        if n_users < 2 or n_items < 2:
            self.fitted = False
            return self

        max_components = min(n_users - 1, n_items - 1)
        n_components = int(max(1, min(self.n_factors, max_components)))
        if n_components < 1:
            self.fitted = False
            return self

        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.U = self.svd.fit_transform(matrix)
        self.V = self.svd.components_
        self.item_ids = list(item_ids)
        self.user_ids = list(user_ids)
        self.fitted = True
        return self

    def predict_scores(self, user_idx: int) -> np.ndarray:
        if not self.fitted or self.U is None or self.V is None:
            return np.array([], dtype=np.float32)
        return (self.U[user_idx] @ self.V).astype(np.float32)

    def normalized_scores_for_user(self, user_id: str) -> dict[str, float]:
        if not self.fitted or user_id not in self.user_ids:
            return {}
        idx = self.user_ids.index(user_id)
        raw = self.predict_scores(idx)
        if raw.size == 0:
            return {}

        r_min = float(np.min(raw))
        r_max = float(np.max(raw))
        if r_max - r_min <= 1e-9:
            normalized = np.full_like(raw, fill_value=0.5, dtype=np.float32)
        else:
            normalized = (raw - r_min) / (r_max - r_min)

        return {self.item_ids[i]: float(normalized[i]) for i in range(len(self.item_ids))}
