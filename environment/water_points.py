"""Water point generation and failure-risk scoring.

The features and the status distribution mirror the Tanzania "Pump it Up"
water point dataset (Taarifa / Tanzanian Ministry of Water) used in the
capstone classification project:

* status_group: functional 54.3 %, functional needs repair 7.3 %, non functional 38.4 %
* extraction_type_class: gravity, handpump, submersible, motorpump, other
* payment_type: never pay, pay per bucket, monthly, annually, on failure
* construction age, population served

The risk score is the daily failure propensity of each water point. By default
it comes from a logistic model whose coefficients follow the direction of the
effects found in the capstone classifier (older pumps, motorised extraction
and "never pay" management fail more often). To use the real classifier,
export a CSV with columns ``x_km,y_km,population,risk`` (risk = predicted
probability of being non functional) and pass ``points_csv=...`` to the
environment.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Pump it Up status distribution (training labels, 59,400 rows)
STATUS_PROBS = np.array([0.543, 0.073, 0.384])  # functional, needs repair, non functional

EXTRACTION_TYPES = ["gravity", "handpump", "submersible", "motorpump", "other"]
EXTRACTION_PROBS = [0.45, 0.28, 0.10, 0.05, 0.12]
EXTRACTION_COEF = {"gravity": -0.5, "handpump": 0.0, "submersible": 0.3, "motorpump": 0.6, "other": 0.9}

PAYMENT_TYPES = ["never pay", "pay per bucket", "monthly", "annually", "on failure"]
PAYMENT_PROBS = [0.43, 0.15, 0.14, 0.06, 0.22]
PAYMENT_COEF = {"never pay": 0.6, "pay per bucket": -0.3, "monthly": -0.5, "annually": -0.2, "on failure": 0.3}

AGE_COEF = 0.05
INTERCEPT = -1.6


@dataclass
class WaterPoint:
    x_km: float
    y_km: float
    population: int
    risk: float
    age_years: int = 0
    extraction_type: str = "unknown"
    payment_type: str = "unknown"


def risk_score(age_years: float, extraction_type: str, payment_type: str) -> float:
    """Logistic failure-risk score in [0, 1]."""
    z = (
        INTERCEPT
        + AGE_COEF * age_years
        + EXTRACTION_COEF.get(extraction_type, 0.0)
        + PAYMENT_COEF.get(payment_type, 0.0)
    )
    return float(1.0 / (1.0 + np.exp(-z)))


def generate_points(n_points: int, map_size_km: float, rng: np.random.Generator) -> list[WaterPoint]:
    """Sample a district of water points with Pump-it-Up-like features."""
    points: list[WaterPoint] = []
    centre = map_size_km / 2
    min_gap = map_size_km / (2.2 * np.sqrt(n_points))
    while len(points) < n_points:
        x, y = rng.uniform(0.08 * map_size_km, 0.92 * map_size_km, size=2)
        # keep points apart from each other and from the depot (map centre)
        if np.hypot(x - centre, y - centre) < min_gap:
            continue
        if any(np.hypot(x - p.x_km, y - p.y_km) < min_gap for p in points):
            continue
        age = int(rng.integers(0, 40))
        extraction = str(rng.choice(EXTRACTION_TYPES, p=EXTRACTION_PROBS))
        payment = str(rng.choice(PAYMENT_TYPES, p=PAYMENT_PROBS))
        # population served is heavily right-skewed in the dataset
        population = int(np.clip(rng.lognormal(mean=6.6, sigma=0.7), 100, 4000))
        points.append(
            WaterPoint(
                x_km=float(x),
                y_km=float(y),
                population=population,
                risk=risk_score(age, extraction, payment),
                age_years=age,
                extraction_type=extraction,
                payment_type=payment,
            )
        )
    return points


def load_points_csv(path: str | Path) -> list[WaterPoint]:
    """Load water points (e.g. with risk predicted by the capstone classifier)."""
    points = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            points.append(
                WaterPoint(
                    x_km=float(row["x_km"]),
                    y_km=float(row["y_km"]),
                    population=int(float(row["population"])),
                    risk=float(row["risk"]),
                    age_years=int(float(row.get("age_years", 0) or 0)),
                    extraction_type=row.get("extraction_type", "unknown") or "unknown",
                    payment_type=row.get("payment_type", "unknown") or "unknown",
                )
            )
    return points
