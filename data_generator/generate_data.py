#!/usr/bin/env python3
"""
Synthetic data generator for the BaaS (Banking-as-a-Service) lakehouse project.

EVERYTHING HERE IS FAKE. Partner names, customers, accounts and transactions are
randomly generated. No real company data is used.

Output layout (this is what you upload to the ADLS Gen2 `landing` folder):

    landing/
      reference/partners.csv, customers.csv, accounts.csv, fx_rates.csv
      transactions/ingest_date=YYYY-MM-DD/transactions_YYYYMMDD.csv
      _labels/fraud_txn_ids.csv          <- ground truth, only for validating the fraud rule

Deliberately injected data problems (so the Silver layer has real work to do):
    * ~0.5% duplicate rows (at-least-once delivery, replayed next day)
    * ~0.1% missing account_id
    * ~1%   lower-case currency codes
    * ~2%   timestamps in a second format (dd/MM/yyyy HH:mm:ss)
    * ~0.05% negative amounts on non-reversal rows
    * ~1%   late-arriving rows (land 1-3 days after the event)
    * Skewed account activity (a few "hot" accounts) -> data skew to talk about
    * Fraud-like bursts: many high-value cross-border / cash-out txns in 10 minutes

Usage:
    python generate_data.py --out landing --n-txn 3000000
    python generate_data.py --out landing_small --n-txn 200000 --n-customers 20000 --days 60
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

COUNTRY_CCY = {"SM": "EUR", "IT": "EUR", "GB": "GBP", "EG": "EGP", "KE": "KES", "NG": "NGN", "AE": "AED"}
COUNTRIES = list(COUNTRY_CCY)

CCYS = ["EUR", "GBP", "USD", "EGP", "KES", "NGN", "AED"]
FX_BASE = np.array([1.0, 1.17, 0.92, 0.019, 0.0070, 0.00060, 0.25])      # approx rate_to_eur
FX_DRIFT = np.array([0, 0, 0, -0.0003, -0.0002, -0.0006, 0.0])            # daily log drift
FX_VOL = np.array([0, 0.002, 0.003, 0.004, 0.004, 0.006, 0.0005])         # daily log volatility

# Fictional BaaS partners (clients of the platform)
PARTNERS = [
    ("P001", "Titano Pay", "SM", "neobank"),
    ("P002", "Adriatic Cards", "IT", "card_issuer"),
    ("P003", "Thames Neo", "GB", "neobank"),
    ("P004", "Nile Wallet", "EG", "wallet"),
    ("P005", "Cairo Remit", "EG", "remittance"),
    ("P006", "Savanna Money", "KE", "wallet"),
    ("P007", "Lagos Link", "NG", "neobank"),
    ("P008", "Gulf Spend", "AE", "card_issuer"),
    ("P009", "Tiber Bank", "IT", "bank"),
    ("P010", "Delta Commerce", "EG", "merchant_platform"),
]
PARTNER_W = [0.06, 0.10, 0.12, 0.18, 0.10, 0.12, 0.12, 0.08, 0.04, 0.08]

TXN_TYPES = ["card_payment", "bank_transfer", "cross_border_transfer", "wallet_topup", "cash_out", "fee"]
TYPE_P = [0.40, 0.20, 0.10, 0.15, 0.10, 0.05]
MU = np.array([3.0, 5.0, 5.6, 3.8, 4.2, 0.5])      # lognormal mean of amount in EUR
SIG = np.array([0.9, 1.1, 1.0, 0.8, 0.7, 0.5])
CROSS_IDX = TXN_TYPES.index("cross_border_transfer")
CASHOUT_IDX = TXN_TYPES.index("cash_out")

CHANNELS = ["mobile_app", "api", "pos", "atm", "web"]
CH_P = [0.38, 0.22, 0.25, 0.05, 0.10]
MCC = ["grocery", "fuel", "restaurants", "e-commerce", "travel", "utilities", "entertainment", "other"]
MCC_P = [0.24, 0.10, 0.14, 0.22, 0.06, 0.08, 0.06, 0.10]


def ids(prefix, start, count, width):
    return np.char.add(prefix, np.char.zfill(np.arange(start, start + count).astype(str), width))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="landing")
    ap.add_argument("--n-txn", type=int, default=3_000_000)
    ap.add_argument("--n-customers", type=int, default=200_000)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    start_ts = pd.Timestamp(args.start)
    dates = pd.date_range(start_ts, periods=args.days)
    ref_dir = os.path.join(args.out, "reference")
    os.makedirs(ref_dir, exist_ok=True)

    # ---------------- reference data ----------------
    pd.DataFrame(PARTNERS, columns=["partner_id", "partner_name", "country", "segment"]).to_csv(
        f"{ref_dir}/partners.csv", index=False)

    n_c = args.n_customers
    p_idx = rng.choice(len(PARTNERS), n_c, p=PARTNER_W)
    c_country = np.array([PARTNERS[i][2] for i in p_idx])
    customers = pd.DataFrame({
        "customer_id": ids("C", 0, n_c, 8),
        "partner_id": [PARTNERS[i][0] for i in p_idx],
        "country": c_country,
        "kyc_status": rng.choice(["verified", "pending", "rejected"], n_c, p=[0.86, 0.10, 0.04]),
        "risk_tier": rng.choice(["low", "medium", "high"], n_c, p=[0.72, 0.21, 0.07]),
        "created_at": (start_ts - pd.to_timedelta(rng.integers(30, 900, n_c), unit="D")).strftime("%Y-%m-%d"),
    })
    customers.to_csv(f"{ref_dir}/customers.csv", index=False)

    n_a = int(n_c * 1.3)
    owner = rng.integers(0, n_c, n_a)
    owner[:n_c] = np.arange(n_c)                       # every customer has >= 1 account
    acct_country = c_country[owner]
    local_ccy = pd.Series(acct_country).map(COUNTRY_CCY).to_numpy()
    acct_ccy = np.where(rng.random(n_a) < 0.8, local_ccy, rng.choice(["EUR", "USD"], n_a))
    acct_ids = ids("A", 0, n_a, 9)
    accounts = pd.DataFrame({
        "account_id": acct_ids,
        "customer_id": customers["customer_id"].to_numpy()[owner],
        "currency": acct_ccy,
        "account_type": rng.choice(["current", "wallet", "card"], n_a, p=[0.50, 0.35, 0.15]),
        "status": rng.choice(["active", "dormant", "closed"], n_a, p=[0.93, 0.05, 0.02]),
        "opened_at": (start_ts - pd.to_timedelta(rng.integers(10, 800, n_a), unit="D")).strftime("%Y-%m-%d"),
    })
    accounts.to_csv(f"{ref_dir}/accounts.csv", index=False)
    ccy_idx = pd.Series(acct_ccy).map({c: i for i, c in enumerate(CCYS)}).to_numpy()

    fx = FX_BASE * np.exp(np.cumsum(FX_DRIFT + FX_VOL * rng.standard_normal((args.days, len(CCYS))), axis=0))
    pd.DataFrame({
        "date": np.repeat(dates.strftime("%Y-%m-%d"), len(CCYS)),
        "currency": np.tile(CCYS, args.days),
        "rate_to_eur": fx.reshape(-1).round(8),
    }).to_csv(f"{ref_dir}/fx_rates.csv", index=False)

    # ---------------- transactions ----------------
    def build(acc_idx, day, sec, tidx, amt_eur, status, id_start):
        m = len(acc_idx)
        ts = (start_ts + pd.to_timedelta(day, unit="D") + pd.to_timedelta(sec, unit="s")
              ).strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()
        rate = fx[day, ccy_idx[acc_idx]]
        types = np.array(TXN_TYPES)[tidx]
        country = acct_country[acc_idx]
        cp = np.where(types == "cross_border_transfer", rng.choice(COUNTRIES, m), country)
        mcc = np.where(types == "card_payment", rng.choice(MCC, m, p=MCC_P), None)
        delay = rng.choice([0, 1, 2, 3], m, p=[0.99, 0.006, 0.003, 0.001])
        return pd.DataFrame({
            "txn_id": ids("T", id_start, m, 10),
            "account_id": acct_ids[acc_idx],
            "txn_ts": ts,
            "txn_type": types,
            "amount": np.maximum(np.round(amt_eur / rate, 2), 0.01),
            "currency": acct_ccy[acc_idx],
            "status": status,
            "channel": rng.choice(CHANNELS, m, p=CH_P),
            "merchant_category": mcc,
            "counterparty_country": cp,
            "landing_date": np.minimum(day + delay, args.days - 1),
        })

    n = args.n_txn
    dow = dates.dayofweek.to_numpy()
    w = (np.linspace(0.8, 1.3, args.days)
         * np.where(dow == 5, 1.15, np.where(dow == 6, 0.9, 1.0))
         * np.where(dates.day.to_numpy() >= 28, 1.2, 1.0))
    w = w / w.sum()
    day_idx = rng.choice(args.days, n, p=w)

    h = np.arange(24)
    hp = np.exp(-(((h - 15) / 5.0) ** 2)) + 0.05
    hp = hp / hp.sum()
    sec = rng.choice(24, n, p=hp) * 3600 + rng.integers(0, 3600, n)

    acc_idx = (rng.random(n) ** 1.6 * n_a).astype(int)          # skewed -> hot accounts
    tidx = rng.choice(len(TXN_TYPES), n, p=TYPE_P)
    amt_eur = rng.lognormal(MU[tidx], SIG[tidx])
    status = rng.choice(["settled", "pending", "failed", "reversed"], n, p=[0.93, 0.03, 0.03, 0.01])
    normal = build(acc_idx, day_idx, sec, tidx, amt_eur, status, 0)

    # fraud-like bursts: 6-14 high-value cross-border / cash-out txns within 10 minutes
    n_burst_acc = max(50, n // 10000)
    sizes = rng.integers(6, 15, n_burst_acc)
    total = int(sizes.sum())
    b_acc = np.repeat(rng.integers(0, n_a, n_burst_acc), sizes)
    b_day = np.repeat(rng.integers(0, args.days, n_burst_acc), sizes)
    b_sec = np.repeat(rng.integers(0, 80000, n_burst_acc), sizes) + rng.integers(0, 600, total)
    b_tidx = rng.choice([CROSS_IDX, CASHOUT_IDX], total)
    b_amt = rng.lognormal(6.3, 0.4, total)
    bursts = build(b_acc, b_day, b_sec, b_tidx, b_amt, np.full(total, "settled"), n)

    df = pd.concat([normal, bursts], ignore_index=True)
    labels_dir = os.path.join(args.out, "_labels")
    os.makedirs(labels_dir, exist_ok=True)
    bursts[["txn_id"]].to_csv(f"{labels_dir}/fraud_txn_ids.csv", index=False)

    # ---------------- inject dirty data ----------------
    def mask(p):
        return rng.random(len(df)) < p

    df.loc[mask(0.001), "account_id"] = None
    m = mask(0.01)
    df.loc[m, "currency"] = df.loc[m, "currency"].str.lower()
    m = mask(0.0005) & (df["status"] != "reversed").to_numpy()
    df.loc[m, "amount"] = -df.loc[m, "amount"].abs()
    m = mask(0.02)
    df.loc[m, "txn_ts"] = pd.to_datetime(df.loc[m, "txn_ts"], format="%Y-%m-%dT%H:%M:%SZ").dt.strftime(
        "%d/%m/%Y %H:%M:%S")
    dup = df.sample(frac=0.005, random_state=args.seed).copy()
    dup["landing_date"] = np.minimum(dup["landing_date"] + 1, args.days - 1)
    df = pd.concat([df, dup], ignore_index=True)

    # ---------------- write daily landing files ----------------
    files = 0
    for d, g in df.groupby("landing_date"):
        day = dates[int(d)].strftime("%Y-%m-%d")
        folder = os.path.join(args.out, "transactions", f"ingest_date={day}")
        os.makedirs(folder, exist_ok=True)
        g.drop(columns="landing_date").to_csv(
            os.path.join(folder, f"transactions_{day.replace('-', '')}.csv"), index=False)
        files += 1

    print(f"partners={len(PARTNERS)} customers={n_c:,} accounts={n_a:,} fx_rows={args.days * len(CCYS):,}")
    print(f"transaction rows written={len(df):,} (incl. {len(dup):,} duplicates, {total:,} fraud-burst rows)")
    print(f"daily files={files}  output='{args.out}'  time={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
