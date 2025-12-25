import numpy as np
import pandas as pd

def generate_synthetic_access_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    user_role = rng.integers(0, 5, size=n)
    device_trust_score = np.clip(rng.normal(0.7, 0.2, n), 0, 1)
    geo_risk = np.clip(rng.beta(2, 8, n), 0, 1)
    hour_of_day = rng.integers(0, 24, n)
    auth_strength = rng.integers(0, 3, n)
    resource_sensitivity = rng.integers(0, 4, n)
    session_duration_minutes = np.abs(rng.normal(30, 20, n))
    access_history_risk = np.clip(rng.beta(1.5, 6, n), 0, 1)

    df = pd.DataFrame({
        'user_role': user_role,
        'device_trust_score': device_trust_score,
        'geo_risk': geo_risk,
        'hour_of_day': hour_of_day,
        'auth_strength': auth_strength,
        'resource_sensitivity': resource_sensitivity,
        'session_duration_minutes': session_duration_minutes,
        'access_history_risk': access_history_risk,
    })

    # Non-linear scoring model
    score = (
        2.5 * df["device_trust_score"]
        + 1.5 * (df["auth_strength"] / 2)
        - 2.0 * df["geo_risk"]
        - 1.0 * (df["resource_sensitivity"] / 3)
        - 1.5 * df["access_history_risk"]
        - 0.01 * df["session_duration_minutes"]
    )

    # Interaction term
    late = ((df["hour_of_day"] < 6) | (df["hour_of_day"] > 22)).astype(int)
    score -= 1.5 * late * (df["resource_sensitivity"] / 3) * (1 - df["auth_strength"] / 2)

    # Role + geo interaction
    score -= 0.8 * df["geo_risk"] * (df["user_role"] >= 3).astype(int)

    # Slight sinusoidal pattern
    score += 0.5 * np.sin(df["hour_of_day"] / 24 * 2 * np.pi)

    prob = 1/(1 + np.exp(-score))
    # Deterministic labels for high-accuracy modeling
    labels = (prob > 0.5).astype(int)
    df["label"] = labels
    return df