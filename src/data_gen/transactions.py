"""Card transactions. One row per purchase or cash withdrawal.

About 0.5% of rows are fraud. The fraud is not random noise: it is built as
the patterns a fraud strategy later looks for.

- Burst: several small tickets in a few minutes (card testing).
- Foreign + new device: a country the customer does not live in, on a phone
  the customer has not used before.
- Night high value: a large ticket between midnight and 5am.
- Amount spike: a ticket about 10x that customer's own typical spend.
- Stealth: a fraud label on an otherwise ordinary purchase. These exist so a
  rulebook cannot catch every fraud row.

A few legitimate "busy hour" grocery runs are also written, so a velocity
rule will have some false positives. That is what happens on a festival weekend.

Trade-off: labels are created from these patterns, so a rule that copies the
pattern will look strong. The stealth slice and the legitimate busy hours
stop it being a perfect score. Phase 3 should still measure AUC on a holdout,
not celebrate that the rules rediscover the generator.

Feature timing, for the SQL that comes later
---------------------------------------------
"Amount versus the customer's 30-day average" must be computed from *earlier*
transactions only. If the current ticket is included in its own average, one
huge purchase pulls the average up and the spike looks milder than it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_gen.config import (
    CARDHOLDER_FRACTION,
    DEFAULT_N_TRANSACTIONS,
    FRAUD_COUNTRIES,
    FRAUD_RATE,
    FRAUD_SHARES,
    HOME_COUNTRY,
    TRANSACTION_COLUMNS,
    TRAVEL_COUNTRIES,
    DataGenConfig,
    legit_hour_weights,
)
from src.data_gen.utils import round_money, split_budget

MERCHANT_CATEGORIES = (
    "grocery",
    "fuel",
    "restaurant",
    "ecommerce",
    "travel",
    "atm",
    "electronics",
    "utilities",
    "jewellery",
    "healthcare",
)
MERCHANT_P = (0.22, 0.12, 0.14, 0.16, 0.06, 0.08, 0.05, 0.09, 0.02, 0.06)

# (ln median amount, spread, cap in INR). Caps stop a normal grocery row looking like fraud.
_AMOUNT_LOG: dict[str, tuple[float, float, float]] = {
    "grocery": (6.10, 0.45, 8_000),
    "fuel": (7.00, 0.35, 8_000),
    "restaurant": (6.40, 0.50, 12_000),
    "ecommerce": (6.90, 0.70, 40_000),
    "travel": (8.40, 0.60, 80_000),
    "atm": (8.20, 0.40, 20_000),
    "electronics": (8.60, 0.55, 150_000),
    "utilities": (7.20, 0.30, 15_000),
    "jewellery": (9.40, 0.50, 250_000),
    "healthcare": (7.50, 0.55, 40_000),
}


def generate_transactions(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    cfg: DataGenConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Clean transaction file plus a few diagnostics (pattern counts).

    The diagnostics are not columns in the file. A real feed does not label
    which typology a fraudster used; we only keep the counts so the run
    report can say how the 0.5% was built.
    """

    card_ids = _cardholders(rng, customers, cfg)
    legit = _legitimate(rng, card_ids, cfg)
    _rewrite_busy_hours(rng, legit)
    fraud = _fraudulent(rng, legit, card_ids, cfg)

    frame = pd.concat([legit, fraud], ignore_index=True)
    frame = frame.sort_values(["_ts", "customer_id"]).reset_index(drop=True)
    diagnostics: dict[str, object] = {
        "fraud_patterns": (
            frame.loc[frame["is_fraud"] == 1, "_pattern"].value_counts().astype(int).to_dict()
        ),
        "legit_busy_hour_txns": int((frame["_pattern"] == "legit_busy_hour").sum()),
    }
    frame["txn_id"] = [f"TXN{index:010d}" for index in range(1, len(frame) + 1)]
    frame["txn_ts"] = pd.to_datetime(frame["_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    frame["amount"] = round_money(frame["amount"].to_numpy())
    public = frame.loc[:, list(TRANSACTION_COLUMNS)].copy()
    _check(public, customers, cfg)
    return public, diagnostics


def _cardholders(rng: np.random.Generator, customers: pd.DataFrame, cfg: DataGenConfig) -> np.ndarray:
    n_card = int(round(CARDHOLDER_FRACTION * cfg.n_customers))
    n_card = min(max(n_card, 1), cfg.n_customers)
    picked = rng.choice(len(customers), size=n_card, replace=False)
    return customers["customer_id"].to_numpy()[picked]


def _legitimate(
    rng: np.random.Generator,
    card_ids: np.ndarray,
    cfg: DataGenConfig,
) -> pd.DataFrame:
    n_legit = cfg.n_transactions - cfg.n_fraud
    # Lognormal activity: many quiet cards, a few very active ones.
    activity = rng.lognormal(0.0, 0.85, len(card_ids))
    counts = rng.multinomial(n_legit, activity / activity.sum())
    customer_id = np.repeat(card_ids, counts)
    n = len(customer_id)

    merchant = rng.choice(list(MERCHANT_CATEGORIES), size=n, p=MERCHANT_P)
    day = rng.integers(0, cfg.transaction_days, n)
    hour = rng.choice(24, size=n, p=legit_hour_weights())
    minute = rng.integers(0, 60, n)
    second = rng.integers(0, 60, n)

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "amount": _amounts(rng, merchant),
            "merchant_category": merchant,
            "country": _travel_countries(rng, customer_id, day, card_ids, cfg),
            "device_id": _known_devices(rng, customer_id),
            "_ts": _stamps(cfg.transaction_start, day, hour, minute, second),
            "is_fraud": np.zeros(n, dtype=int),
            "_pattern": "",
        }
    )


def _amounts(rng: np.random.Generator, merchant: np.ndarray) -> np.ndarray:
    amount = np.empty(len(merchant))
    for name, (mu, sigma, cap) in _AMOUNT_LOG.items():
        mask = merchant == name
        if not mask.any():
            continue
        drawn = rng.lognormal(mu, sigma, int(mask.sum()))
        amount[mask] = np.clip(drawn, 1.0, cap)
    return round_money(amount)


def _known_devices(rng: np.random.Generator, customer_id: np.ndarray) -> np.ndarray:
    """Most spend is on the customer's own phone. A few rows are a new handset.

    Suffix A is the primary device, B is a second known device, N... is a
    one-off upgrade. Fraud devices use the suffix F, so they do not collide.
    """

    n = len(customer_id)
    roll = rng.random(n)
    device = np.empty(n, dtype=object)
    primary = roll < 0.88
    secondary = (roll >= 0.88) & (roll < 0.985)
    upgrade = roll >= 0.985
    device[primary] = [f"DEV{cid[4:]}A" for cid in customer_id[primary]]
    device[secondary] = [f"DEV{cid[4:]}B" for cid in customer_id[secondary]]
    upgrade_at = np.flatnonzero(upgrade)
    device[upgrade] = [f"DEV{customer_id[index][4:]}N{index:07d}" for index in upgrade_at]
    return device


def _travel_countries(
    rng: np.random.Generator,
    customer_id: np.ndarray,
    day: np.ndarray,
    card_ids: np.ndarray,
    cfg: DataGenConfig,
) -> np.ndarray:
    """A small set of customers take a 4-day trip. Their foreign tickets are legitimate.

    Fraud abroad is a single out-of-place ticket, not a trip. Both will set a
    "country changed" flag in SQL; that false positive on real travel is intended.
    """

    n_travelers = min(len(card_ids), max(1, int(round(0.05 * len(card_ids)))))
    travelers = rng.choice(card_ids, size=n_travelers, replace=False)
    latest_start = max(cfg.transaction_days - 4, 1)
    start_map = pd.Series(rng.integers(0, latest_start, size=n_travelers), index=travelers)
    dest_map = pd.Series(rng.choice(list(TRAVEL_COUNTRIES), size=n_travelers), index=travelers)
    start = start_map.reindex(customer_id).to_numpy()
    dest = dest_map.reindex(customer_id).to_numpy()
    on_trip = (day >= start) & (day <= start + 3)
    country = np.full(len(customer_id), HOME_COUNTRY, dtype=object)
    country[on_trip] = dest[on_trip]
    return country


def _rewrite_busy_hours(rng: np.random.Generator, legit: pd.DataFrame) -> None:
    """Cluster a few existing legitimate tickets into a 20-minute grocery run.

    Rewriting (instead of adding rows) keeps the file at the requested size.
    On the full portfolio this is about 40 customers. Smaller runs get fewer.
    """

    full_legit = int(round(DEFAULT_N_TRANSACTIONS * (1.0 - FRAUD_RATE)))
    n_groups = max(1, int(round(40 * len(legit) / full_legit)))
    counts = legit.groupby("customer_id").size()
    eligible = counts[counts >= 6].index.to_numpy()
    if len(eligible) == 0:
        return
    chosen = rng.choice(eligible, size=min(n_groups, len(eligible)), replace=False)
    grouped = legit.groupby("customer_id").groups
    for customer_id in chosen:
        indexes = np.array(list(grouped[customer_id])[:5])
        anchor = legit.at[indexes[0], "_ts"]
        for step, index in enumerate(indexes):
            legit.at[index, "_ts"] = anchor + pd.Timedelta(minutes=int(step * 4))
            legit.at[index, "merchant_category"] = "grocery"
            legit.at[index, "country"] = HOME_COUNTRY
            legit.at[index, "amount"] = round(float(rng.uniform(180, 1_400)), 2)
            legit.at[index, "_pattern"] = "legit_busy_hour"
            legit.at[index, "is_fraud"] = 0


def _fraudulent(
    rng: np.random.Generator,
    legit: pd.DataFrame,
    card_ids: np.ndarray,
    cfg: DataGenConfig,
) -> pd.DataFrame:
    budget = split_budget(cfg.n_fraud, FRAUD_SHARES)
    # A "burst" of one or two tickets is not a burst. Fold those rows into stealth.
    if budget["fraud_burst"] < 4:
        budget["fraud_stealth"] += budget["fraud_burst"]
        budget["fraud_burst"] = 0

    pool = list(card_ids)
    parts: list[pd.DataFrame] = []
    if budget["fraud_burst"]:
        parts.append(_bursts(rng, budget["fraud_burst"], pool, cfg))
    if budget["fraud_foreign_new_device"]:
        parts.append(_foreign_new_device(rng, budget["fraud_foreign_new_device"], pool, cfg))
    if budget["fraud_night_high_value"]:
        parts.append(_night_high_value(rng, budget["fraud_night_high_value"], pool, cfg))
    if budget["fraud_amount_spike"]:
        parts.append(_amount_spikes(rng, legit, budget["fraud_amount_spike"], pool, cfg))
    if budget["fraud_stealth"]:
        parts.append(_stealth(rng, legit, budget["fraud_stealth"], pool, cfg))

    fraud = pd.concat(parts, ignore_index=True)
    if len(fraud) != cfg.n_fraud:
        raise RuntimeError(f"expected {cfg.n_fraud} fraud rows, built {len(fraud)}")
    return fraud


def _take_customers(rng: np.random.Generator, pool: list[str], k: int) -> np.ndarray:
    """Draw cardholders without putting the same person in every pattern."""

    if k > len(pool):
        # A tiny practice file can run out of unique cardholders. Reuse is
        # allowed then; the full portfolio has far more cardholders than patterns.
        return rng.choice(np.array(pool), size=k, replace=True)
    chosen = rng.choice(np.array(pool), size=k, replace=False)
    chosen_set = set(chosen.tolist())
    pool[:] = [customer_id for customer_id in pool if customer_id not in chosen_set]
    return chosen


def _bursts(
    rng: np.random.Generator,
    n_rows: int,
    pool: list[str],
    cfg: DataGenConfig,
) -> pd.DataFrame:
    sizes = _episode_sizes(n_rows, rng)
    customers = _take_customers(rng, pool, len(sizes))
    rows: list[dict[str, object]] = []
    for episode, (customer_id, size) in enumerate(zip(customers, sizes, strict=True)):
        anchor = _one_stamp(rng, cfg, hour_low=8, hour_high=22)
        if size > 1:
            gaps = np.concatenate([[0], np.cumsum(rng.integers(2, 6, size - 1))])
        else:
            gaps = np.array([0])
        base = float(rng.uniform(49, 220))
        category = "ecommerce" if episode % 2 == 0 else "atm"
        for step in range(size):
            rows.append(
                {
                    "customer_id": customer_id,
                    "amount": round(base + float(rng.uniform(0, 40)), 2),
                    "merchant_category": category,
                    "country": HOME_COUNTRY,
                    "device_id": f"DEV{customer_id[4:]}F{episode:05d}",
                    "_ts": anchor + pd.Timedelta(minutes=int(gaps[step])),
                    "is_fraud": 1,
                    "_pattern": "fraud_burst",
                }
            )
    return pd.DataFrame(rows)


def _episode_sizes(n_rows: int, rng: np.random.Generator) -> list[int]:
    """Split a row budget into bursts of 5 to 7 tickets, without losing a row."""

    sizes: list[int] = []
    left = n_rows
    while left > 0:
        if left < 8:
            sizes.append(left)
            break
        size = int(rng.integers(5, 8))
        if left - size < 5:
            size = left
        sizes.append(size)
        left -= size
    return sizes


def _foreign_new_device(
    rng: np.random.Generator,
    n_rows: int,
    pool: list[str],
    cfg: DataGenConfig,
) -> pd.DataFrame:
    customers = _take_customers(rng, pool, n_rows)
    return _simple_fraud(
        customers=customers,
        stamps=_many_stamps(rng, cfg, n_rows, 0, 24),
        amount=np.clip(rng.lognormal(8.3, 0.45, n_rows), 500, 80_000),
        merchant=rng.choice(["ecommerce", "travel", "electronics", "atm"], size=n_rows),
        country=rng.choice(list(FRAUD_COUNTRIES), size=n_rows),
        device=np.array([f"DEV{cid[4:]}F9{index:04d}" for index, cid in enumerate(customers)]),
        pattern="fraud_foreign_new_device",
    )


def _night_high_value(
    rng: np.random.Generator,
    n_rows: int,
    pool: list[str],
    cfg: DataGenConfig,
) -> pd.DataFrame:
    """Large tickets from midnight up to (not including) 5am, on a known device."""

    customers = _take_customers(rng, pool, n_rows)
    amount = np.clip(rng.lognormal(10.6, 0.30, n_rows), 20_000, 200_000)
    return _simple_fraud(
        customers=customers,
        stamps=_many_stamps(rng, cfg, n_rows, 0, 5),
        amount=amount,
        merchant=rng.choice(["electronics", "jewellery", "atm"], size=n_rows),
        country=np.full(n_rows, HOME_COUNTRY, dtype=object),
        device=np.array([f"DEV{cid[4:]}A" for cid in customers]),
        pattern="fraud_night_high_value",
    )


def _amount_spikes(
    rng: np.random.Generator,
    legit: pd.DataFrame,
    n_rows: int,
    pool: list[str],
    cfg: DataGenConfig,
) -> pd.DataFrame:
    """About 10x the customer's own median ticket, in daylight, on their own device.

    Ten times the median is used, rather than exactly five, because the
    customer's average is higher than the median when spend is skewed. The
    later rule "more than 5x the prior 30-day average" should still fire.
    """

    customers, medians = _customers_with_history(rng, legit, pool, n_rows, minimum_txns=8)
    amount = np.maximum(np.round(10.0 * medians, 2), 15_000)
    return _simple_fraud(
        customers=customers,
        stamps=_many_stamps(rng, cfg, n_rows, 10, 20),
        amount=amount,
        merchant=rng.choice(["electronics", "travel", "jewellery"], size=n_rows),
        country=np.full(n_rows, HOME_COUNTRY, dtype=object),
        device=np.array([f"DEV{cid[4:]}A" for cid in customers]),
        pattern="fraud_amount_spike",
    )


def _stealth(
    rng: np.random.Generator,
    legit: pd.DataFrame,
    n_rows: int,
    pool: list[str],
    cfg: DataGenConfig,
) -> pd.DataFrame:
    """Fraud that looks like a normal grocery or fuel purchase.

    Rules based on velocity, country, device, night, or a 5x spike will miss
    these. They are the residual a model has to live with.
    """

    customers, medians = _customers_with_history(rng, legit, pool, n_rows, minimum_txns=3)
    scale = rng.uniform(0.8, 1.8, n_rows)
    hour = rng.choice(24, size=n_rows, p=legit_hour_weights())
    day = _history_days(rng, cfg, n_rows)
    minute = rng.integers(0, 60, n_rows)
    second = rng.integers(0, 60, n_rows)
    return _simple_fraud(
        customers=customers,
        stamps=_stamps(cfg.transaction_start, day, hour, minute, second),
        amount=np.maximum(np.round(medians * scale, 2), 50),
        merchant=rng.choice(["grocery", "fuel", "restaurant"], size=n_rows),
        country=np.full(n_rows, HOME_COUNTRY, dtype=object),
        device=np.array([f"DEV{cid[4:]}A" for cid in customers]),
        pattern="fraud_stealth",
    )


def _customers_with_history(
    rng: np.random.Generator,
    legit: pd.DataFrame,
    pool: list[str],
    k: int,
    minimum_txns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Prefer customers who already have enough legitimate tickets for an average."""

    counts = legit.groupby("customer_id").size()
    medians = legit.groupby("customer_id")["amount"].median()
    pool_set = set(pool)
    chosen: np.ndarray | None = None
    for threshold in (minimum_txns, 3, 1):
        eligible = [
            customer_id
            for customer_id in medians.index
            if customer_id in pool_set and int(counts[customer_id]) >= threshold
        ]
        if len(eligible) >= k:
            chosen = rng.choice(np.array(eligible, dtype=object), size=k, replace=False)
            break
    if chosen is None:
        eligible = medians.index.to_numpy()
        chosen = rng.choice(eligible, size=k, replace=True)
    chosen_set = set(chosen.tolist())
    pool[:] = [customer_id for customer_id in pool if customer_id not in chosen_set]
    return chosen, medians.loc[chosen].to_numpy(dtype=float)


def _simple_fraud(
    customers: np.ndarray,
    stamps: pd.Series,
    amount: np.ndarray,
    merchant: np.ndarray,
    country: np.ndarray,
    device: np.ndarray,
    pattern: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": customers,
            "amount": round_money(np.asarray(amount, dtype=float)),
            "merchant_category": merchant,
            "country": country,
            "device_id": device,
            "_ts": stamps,
            "is_fraud": 1,
            "_pattern": pattern,
        }
    )


def _stamps(
    start: str,
    day: np.ndarray,
    hour: np.ndarray,
    minute: np.ndarray,
    second: np.ndarray,
) -> pd.Series:
    base = pd.Timestamp(start)
    stamps = (
        base
        + pd.to_timedelta(day, unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
        + pd.to_timedelta(second, unit="s")
    )
    return pd.Series(stamps)


def _many_stamps(
    rng: np.random.Generator,
    cfg: DataGenConfig,
    n: int,
    hour_low: int,
    hour_high: int,
) -> pd.Series:
    day = _history_days(rng, cfg, n)
    hour = rng.integers(hour_low, hour_high, n)
    return _stamps(cfg.transaction_start, day, hour, rng.integers(0, 60, n), rng.integers(0, 60, n))


def _one_stamp(rng: np.random.Generator, cfg: DataGenConfig, hour_low: int, hour_high: int) -> pd.Timestamp:
    return _many_stamps(rng, cfg, 1, hour_low, hour_high).iloc[0]


def _history_days(rng: np.random.Generator, cfg: DataGenConfig, n: int) -> np.ndarray:
    """Days late enough in the window that a 30-day history can exist."""

    day_low = min(40, max(cfg.transaction_days - 2, 0))
    day_high = cfg.transaction_days
    if day_high <= day_low:
        day_low = 0
        day_high = max(cfg.transaction_days, 1)
    return rng.integers(day_low, day_high, n)


def _check(frame: pd.DataFrame, customers: pd.DataFrame, cfg: DataGenConfig) -> None:
    if len(frame) != cfg.n_transactions:
        raise RuntimeError(f"expected {cfg.n_transactions} transactions, got {len(frame)}")
    if int(frame["is_fraud"].sum()) != cfg.n_fraud:
        raise RuntimeError("clean fraud row count does not match the configured rate")
    if not frame["txn_id"].is_unique:
        raise RuntimeError("clean transaction ids must be unique")
    if frame["amount"].le(0).any():
        raise RuntimeError("clean transaction amounts must be positive")
    if not set(frame["customer_id"]).issubset(set(customers["customer_id"])):
        raise RuntimeError("clean transactions must point at real customers")
    parsed = pd.to_datetime(frame["txn_ts"], errors="coerce")
    if parsed.isna().any():
        raise RuntimeError("clean transaction timestamps must be real datetimes")
