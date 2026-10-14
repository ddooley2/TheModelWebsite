"""
Ridge Ratings Engine — CFB Model V5

Solves for team offense/defense coefficients + HFA via weighted Ridge
regression with pseudo-observation priors.  All hot paths vectorized.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple


def _sigmoid_shrinkage(k: int, k0: float = 5.0, beta: float = 2.0) -> float:
    """Return in-season weight s_k ∈ (0,1).  k = number of games observed."""
    return 1.0 / (1.0 + np.exp(-(k - k0) / beta))


def _recency_weight(delta_weeks: float, tau: float = 52.0) -> float:
    """Exponential half-life decay weight."""
    return np.exp(-np.log(2) * delta_weeks / tau)


class RidgeRatingsEngine:
    """Weighted Ridge solver for a single rating dimension."""

    def __init__(self, dimension_name: str, ridge_alpha: float = 1.0):
        self.dimension_name = dimension_name
        self.ridge_alpha = ridge_alpha
        self.team_to_idx: Dict[str, int] = {}
        self.coefs_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.hfa_: float = 0.0
        self.coef_se_: Optional[np.ndarray] = None

    def fit(
        self,
        df_obs: pd.DataFrame,
        priors: Optional[Dict[str, Tuple[float, float]]] = None,
        sample_weights: Optional[np.ndarray] = None,
        prior_weight: float = 4.0,
    ):
        # --- build team index ------------------------------------------------
        teams: set = set()
        if not df_obs.empty:
            teams |= set(df_obs["team"]) | set(df_obs["opponent"])
        if priors:
            teams |= set(priors.keys())
        teams_sorted = sorted(teams)
        self.team_to_idx = {t: i for i, t in enumerate(teams_sorted)}
        n_teams = len(teams_sorted)

        if n_teams == 0:
            return

        n_obs = len(df_obs)
        n_cols = 2 * n_teams + 2 # +1 for HFA, +1 for Intercept

        # --- vectorized design matrix ----------------------------------------
        X = np.zeros((n_obs, n_cols))
        y = np.empty(n_obs)
        w = np.ones(n_obs) if sample_weights is None else sample_weights.copy()

        if n_obs > 0:
            teams_arr = df_obs["team"].values
            opps_arr = df_obs["opponent"].values
            home_arr = df_obs["is_home"].values.astype(int)
            y[:] = df_obs["target"].values

            # Vectorized index mapping
            off_indices = np.array([self.team_to_idx[t] for t in teams_arr])
            def_indices = np.array([self.team_to_idx[t] for t in opps_arr])
            row_idx = np.arange(n_obs)

            X[row_idx, off_indices] = 1.0
            X[row_idx, n_teams + def_indices] = 1.0
            home_mask = home_arr == 1
            X[home_mask, -2] = 1.0 # HFA
            X[row_idx, -1] = 1.0   # Intercept

        # --- pseudo-observation rows for priors ------------------------------
        sqrt_pw = np.sqrt(prior_weight)
        n_pseudo = 2 * n_teams
        pseudo_X = np.zeros((n_pseudo, n_cols))
        pseudo_y = np.zeros(n_pseudo)
        pseudo_w = np.ones(n_pseudo)

        for i, team in enumerate(teams_sorted):
            idx = i  # self.team_to_idx[team] == i by construction
            p_off, p_def = (0.0, 0.0) if not priors or team not in priors else priors[team]

            pseudo_X[2 * i, idx] = sqrt_pw
            pseudo_y[2 * i] = sqrt_pw * p_off

            pseudo_X[2 * i + 1, n_teams + idx] = sqrt_pw
            pseudo_y[2 * i + 1] = sqrt_pw * p_def

        X_full = np.vstack([X, pseudo_X])
        y_full = np.concatenate([y, pseudo_y])
        w_full = np.concatenate([w, pseudo_w])

        # --- weighted Ridge solve --------------------------------------------
        sw = np.sqrt(w_full)
        Xw = X_full * sw[:, None]
        yw = y_full * sw

        XtX = Xw.T @ Xw
        Xty = Xw.T @ yw
        
        penalty = np.ones(n_cols) * self.ridge_alpha
        penalty[-1] = 1e-4 # Tiny penalty for intercept to prevent singularity
        penalty[-2] = 1e-4 # Tiny penalty for HFA
        A = XtX + np.diag(penalty)
        beta = np.linalg.solve(A, Xty)

        self.intercept_ = beta[-1]
        self.coefs_ = beta[:-2]
        self.hfa_ = beta[-2]

        # --- coefficient standard errors ------------------------------------
        residuals = y_full - (X_full @ beta)
        n_eff = w_full.sum()
        sigma2 = np.sum(w_full * residuals ** 2) / max(n_eff - n_cols, 1)
        try:
            A_inv_diag = np.diag(np.linalg.inv(A))
            self.coef_se_ = np.sqrt(np.maximum(sigma2 * A_inv_diag[:-2], 0.0))
        except np.linalg.LinAlgError:
            self.coef_se_ = np.full(2 * n_teams, np.nan)

    def get_ratings(self) -> pd.DataFrame:
        n_teams = len(self.team_to_idx)
        teams = list(self.team_to_idx.keys())
        indices = list(self.team_to_idx.values())

        off_ratings = self.coefs_[indices]
        def_ratings = self.coefs_[np.array(indices) + n_teams]

        if self.coef_se_ is not None:
            off_se = self.coef_se_[indices]
            def_se = self.coef_se_[np.array(indices) + n_teams]
            uncertainty = np.sqrt(off_se ** 2 + def_se ** 2)
        else:
            uncertainty = np.zeros(n_teams)

        return pd.DataFrame({
            "team": teams,
            "offense_rating": off_ratings,
            "defense_rating": def_ratings,
            "rating_uncertainty": uncertainty,
            "intercept": self.intercept_,
            "hfa": self.hfa_,
        })
