import pandas as pd
import numpy as np


def clean_agent_data(txns: pd.DataFrame, agents: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Clean raw mobile-money agent transaction records and enrich them with agent metadata.

    Applies deduplication, text standardisation, type coercion, median imputation
    for missing amounts (with an imputation flag), IQR-based outlier flagging,
    and a left join onto the agent register.

    Parameters
    ----------
    txns : pd.DataFrame
        Raw transactions extract containing txn_id, agent_id, txn_type, amount, and txn_date.
    agents : pd.DataFrame
        Agent register containing agent_id, county, and onboarded date.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        A tuple containing the cleaned, merged DataFrame and a step-by-step audit log.
    """
    log = []
    clean = txns.copy()
    initial_rows = len(clean)

    # 1. Uniqueness: Remove duplicate transaction IDs
    dup_count = clean.duplicated(subset=["txn_id"]).sum()
    clean = clean.drop_duplicates(subset=["txn_id"]).copy()
    log.append(f"Uniqueness: Removed {dup_count} duplicate transactions ({initial_rows} -> {len(clean)} rows).")

    # 2. Consistency: Standardise txn_type to 'Deposit' or 'Withdrawal'
    clean["txn_type"] = clean["txn_type"].astype(str).str.strip().str.title()
    invalid_types = (~clean["txn_type"].isin(["Deposit", "Withdrawal"])).sum()
    log.append(
        f"Consistency: Standardised txn_type into 'Deposit' and 'Withdrawal' (invalid remaining: {invalid_types})."
    )

    # 3. Validity & Types: Convert amount to numeric and txn_date to datetime
    clean["amount"] = pd.to_numeric(
        clean["amount"].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    clean["txn_date"] = pd.to_datetime(
        clean["txn_date"].astype(str).str.replace("/", "-", regex=False).str.strip(),
        format="%Y-%m-%d",
    )
    log.append(
        "Validity: Converted amount to float64 (coercing 'N/A' to NaN) and standardised txn_date separators to datetime64."
    )

    # 4. Completeness: Flag missing amounts first, then impute with median
    clean["amount_imputed"] = clean["amount"].isna()
    missing_amt_count = int(clean["amount_imputed"].sum())
    median_amount = clean["amount"].median()
    clean["amount"] = clean["amount"].fillna(median_amount)
    log.append(
        f"Completeness: Flagged and imputed {missing_amt_count} missing/invalid amounts using median (KES {median_amount:,.2f})."
    )

    # 5. Outliers: Flag extreme values using IQR method on non-imputed observed values
    observed_amounts = clean.loc[~clean["amount_imputed"], "amount"]
    q1 = observed_amounts.quantile(0.25)
    q3 = observed_amounts.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    clean["is_outlier"] = (clean["amount"] < lower_fence) | (clean["amount"] > upper_fence)
    outlier_count = int(clean["is_outlier"].sum())
    log.append(
        f"Accuracy/Outliers: Flagged {outlier_count} outliers outside IQR fences (upper fence = KES {upper_fence:,.2f}); retained for review."
    )

    # 6. Merge & Referential Integrity: Left join with agents register
    pre_merge_rows = len(clean)
    clean = clean.merge(agents[["agent_id", "county"]], on="agent_id", how="left")
    unknown_agents = int(clean["county"].isna().sum())
    clean["county"] = clean["county"].fillna("Unknown")
    log.append(
        f"Referential Integrity: Merged with agent register via left join ({pre_merge_rows} -> {len(clean)} rows); labeled {unknown_agents} transactions with unrecognised agent IDs as 'Unknown'."
    )

    return clean, log