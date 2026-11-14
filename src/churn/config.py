"""Configuration centrale (cas bancaire : attrition client) : tables Unity Catalog et paramètres métier.

Toutes les tables vivent dans <catalog>.<schema> (gouvernance Unity Catalog).
Le catalog/schema est injecté par le bundle (dev / prod) via les paramètres de job.
"""
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Config:
    catalog: str = "workspace"
    schema: str = "attrition_dev"

    # --- Paramètres de la simulation (source "système opérationnel") ---
    sim_start: str = "2024-07-01"
    sim_months: int = 28                 # données du 2024-07-01 au 2026-10-31
    history_cutoff: str = "2026-07-01"   # 1er chargement = historique avant cette date
    drift_from: str = "2026-07-01"       # le batch incrémental (juil.→oct. 2026) contient une dérive
    n_clients: int = 3000
    seed: int = 42

    # --- Définition métier de l'attrition ---
    churn_horizon_days: int = 90         # attrition = clôture de la relation dans les 90 j suivant l'observation

    # --- Gates MLOps ---
    min_auc: float = 0.75                # seuil absolu de mise en production
    max_auc_drop: float = 0.01           # le challenger ne doit pas faire pire que champion - 1 pt
    psi_warn: float = 0.10
    psi_alert: float = 0.25

    # --- Prescriptif ---
    campaign_budget_eur: float = 2000.0

    @property
    def fq(self) -> str:
        return f"{self.catalog}.{self.schema}"

    def t(self, name: str) -> str:
        """Nom complet d'une table : catalog.schema.name"""
        return f"{self.fq}.{name}"

    @property
    def landing_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/landing"

    @property
    def checkpoint_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/checkpoints"

    @property
    def churn_model_name(self) -> str:
        return f"{self.fq}.attrition_classifier"

    @property
    def segment_model_name(self) -> str:
        return f"{self.fq}.client_segmentation"


# Features utilisées par le modèle de churn (contrat d'interface entre gold et ML)
NUMERIC_FEATURES = [
    # activité du compte
    "recency_days", "n_ops_30d", "n_ops_90d", "card_spend_90d", "inflow_90d",
    # domiciliation et fuite vers la concurrence
    "n_salary_90d", "days_since_salary", "ext_transfer_out_90d", "n_ext_transfers_90d",
    # fragilité / irritants
    "fees_90d", "n_complaints_90d", "avg_satisfaction_90d", "n_advisor_contacts_90d",
    # usage digital
    "n_app_logins_30d", "app_trend_30_90",
    # profil et équipement
    "tenure_days", "age", "nb_produits", "has_credit_immo", "has_assurance_vie", "has_livret",
    "revenu_mensuel_declare",
]
CATEGORICAL_FEATURES = ["segment", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Segmentation comportementale (indépendante du risque) : usage, équipement, irritants
SEGMENT_FEATURES = ["n_app_logins_30d", "card_spend_90d", "nb_produits",
                    "revenu_mensuel_declare", "fees_90d", "n_complaints_90d"]

# Features surveillées pour la dérive. tenure_days et age sont exclues : elles croissent
# mécaniquement avec le temps (dérive structurelle attendue, non actionnable).
DRIFT_FEATURES = [f for f in NUMERIC_FEATURES if f not in ("tenure_days", "age")]

# Découpage temporel (dates d'observation = 1er du mois)
TRAIN_SNAPSHOTS = ("2025-07-01", "2025-12-01")   # entraînement
VALID_SNAPSHOTS = ("2026-01-01", "2026-02-01")   # tuning des hyperparamètres (Optuna)
TEST_SNAPSHOTS = ("2026-03-01", "2026-04-01")    # holdout : gate de promotion + baseline monitoring

KEYS = ["client_id", "snapshot_date"]
LABEL = "churn"


def month_starts(start: str, end: str) -> list[date]:
    """Liste des 1ers du mois entre start et end inclus."""
    import pandas as pd
    return [d.date() for d in pd.date_range(start, end, freq="MS")]


def latest_snapshot(max_event_date):
    """Date d'observation la plus récente : 1er du mois suivant le dernier jour observé
    (max = 2026-06-30 -> 2026-07-01)."""
    import datetime as dt
    d = max_event_date + dt.timedelta(days=1)
    return d.replace(day=1)


def in_window(series, window: tuple[str, str]):
    """Masque pandas : snapshot_date dans [début, fin]."""
    import datetime as dt
    lo, hi = (dt.date.fromisoformat(x) for x in window)
    return (series >= lo) & (series <= hi)
