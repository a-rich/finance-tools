import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

app = typer.Typer(no_args_is_help=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


# ============================================================================
# Tax constants
# ============================================================================

# 2026 federal tax brackets for Married Filing Jointly.
#
# Each tuple is:
#     (upper_bound, marginal_rate)
#
# None means there is no upper bound.
#
# Source: IRS 2026 tax rate tables.
IRS_MFJ_BRACKETS = [
    (24_800, 0.10),
    (100_800, 0.12),
    (211_400, 0.22),
    (403_550, 0.24),
    (512_450, 0.32),
    (768_700, 0.35),
    (None, 0.37),
]

IRS_MFJ_STANDARD_DEDUCTION = 32_200


# California's 2026 estimated-tax instructions currently direct taxpayers
# to use the 2025 tax table. These are therefore the latest published
# California Schedule Y brackets as of this script's 2026 tax-year use.
#
# Source: California FTB 2025 Schedule Y.
CA_MFJ_BRACKETS = [
    (22_158, 0.01),
    (52_528, 0.02),
    (82_904, 0.04),
    (115_084, 0.06),
    (145_448, 0.08),
    (742_958, 0.093),
    (891_542, 0.103),
    (1_485_906, 0.113),
    (None, 0.123),
]

CA_MFJ_STANDARD_DEDUCTION = 11_412

# California Behavioral Health Services Tax:
# additional 1% on taxable income above $1,000,000.
CA_BEHAVIORAL_HEALTH_THRESHOLD = 1_000_000
CA_BEHAVIORAL_HEALTH_RATE = 0.01


# ============================================================================
# Generic helpers
# ============================================================================

def _clean_money(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .replace("nan", pd.NA)
        .astype(float)
    )


def _filter_date_range(
    df: pd.DataFrame,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    date_col: str = "Date",
) -> pd.DataFrame:
    """Filter rows to [start_date, end_date] inclusive.

    Assumes date_col has already been forward-filled so every row in a
    transaction group (header + lot rows) shares the same date.
    """
    if start_date is None and end_date is None:
        return df

    parsed = pd.to_datetime(df[date_col], format="%m/%d/%Y")

    mask = pd.Series(True, index=df.index)

    if start_date is not None:
        mask &= parsed >= start_date

    if end_date is not None:
        mask &= parsed <= end_date

    return df[mask]


def _append_total_row(
    df: pd.DataFrame,
    label_col: str = "Date",
    label: str = "TOTAL",
    skip_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Append a totals row summing numeric columns."""
    skip_cols = set(skip_cols or [])

    numeric_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in skip_cols
    ]

    total = {col: df[col].sum() for col in numeric_cols}
    total[label_col] = label

    total_row = pd.DataFrame([total])

    return pd.concat([df, total_row], ignore_index=True)


# ============================================================================
# Progressive tax helpers
# ============================================================================

def _progressive_tax(
    taxable_income: float,
    brackets: list[tuple[Optional[float], float]],
) -> float:
    """Calculate tax using marginal tax brackets.

    taxable_income must already have deductions applied.
    """
    taxable_income = max(0.0, taxable_income)

    tax = 0.0
    lower_bound = 0.0

    for upper_bound, rate in brackets:
        if upper_bound is None:
            tax += max(0.0, taxable_income - lower_bound) * rate
            break

        if taxable_income <= lower_bound:
            break

        amount_in_bracket = (
            min(taxable_income, upper_bound) - lower_bound
        )

        tax += amount_in_bracket * rate
        lower_bound = upper_bound

    return tax


def _federal_income_tax(gross_income: float) -> float:
    """Approximate federal ordinary income tax for MFJ.

    Uses the 2026 standard deduction and 2026 MFJ brackets.
    """
    taxable_income = max(
        0.0,
        gross_income - IRS_MFJ_STANDARD_DEDUCTION,
    )

    return _progressive_tax(
        taxable_income,
        IRS_MFJ_BRACKETS,
    )


def _california_income_tax(gross_income: float) -> float:
    """Approximate California income tax for MFJ.

    Uses the latest published California Schedule Y brackets (2025)
    and the $11,412 MFJ standard deduction.

    Also includes the 1% Behavioral Health Services Tax on taxable
    income above $1,000,000.
    """
    taxable_income = max(
        0.0,
        gross_income - CA_MFJ_STANDARD_DEDUCTION,
    )

    regular_tax = _progressive_tax(
        taxable_income,
        CA_MFJ_BRACKETS,
    )

    behavioral_health_tax = max(
        0.0,
        taxable_income - CA_BEHAVIORAL_HEALTH_THRESHOLD,
    ) * CA_BEHAVIORAL_HEALTH_RATE

    return regular_tax + behavioral_health_tax


# ============================================================================
# Sale command
# ============================================================================

@app.command("sale")
def compute_tax_on_rsu_sale(
    transactions_path: Path = typer.Option(..., exists=True),
    irs_rate: float = typer.Option(0.20),
    ca_rate: float = typer.Option(0.133),
    start_date: Optional[datetime] = typer.Option(
        None,
        formats=["%Y-%m-%d"],
    ),
    end_date: Optional[datetime] = typer.Option(
        None,
        formats=["%Y-%m-%d"],
    ),
):
    df = pd.read_csv(transactions_path)
    df["TxnGroup"] = df["Date"].notna().cumsum()

    money_cols = [
        "Amount",
        "GrossProceeds",
        "TotalCostBasis",
        "RealizedGainLoss",
    ]

    for col in money_cols:
        if col in df.columns:
            df[col] = _clean_money(df[col])

    header_cols = [
        "Date",
        "Action",
        "Symbol",
        "Quantity",
        "Amount",
    ]

    df[header_cols] = df.groupby("TxnGroup")[header_cols].ffill()

    df = _filter_date_range(
        df,
        start_date,
        end_date,
    )

    sales_only = df[df["Action"] == "Sale"]

    summary = (
        sales_only.groupby("TxnGroup")
        .agg(
            Date=("Date", "first"),
            Symbol=("Symbol", "first"),
            SharesSold=("Quantity", "first"),
            SalePrice=("SalePrice", "first"),
            SaleAmount=("Amount", "first"),
            Fees=("FeesAndCommissions", "first"),
            VestPrice=("VestFairMarketValue", "first"),
            TotalCostBasis=("TotalCostBasis", "sum"),
            TotalRealizedGainLoss=("RealizedGainLoss", "sum"),
        )
        .reset_index(drop=True)
    )

    summary["EstimatedIRSTax"] = (
        summary["TotalRealizedGainLoss"] * irs_rate
    ).round(2)

    summary["EstimatedCATax"] = (
        summary["TotalRealizedGainLoss"] * ca_rate
    ).round(2)

    summary["EstimatedTotalTax"] = (
        summary["EstimatedIRSTax"]
        + summary["EstimatedCATax"]
    )

    summary = _append_total_row(
        summary,
        skip_cols=["SalePrice", "VestPrice"],
    )

    print(summary)

    non_long_term = sales_only[
        sales_only["HoldingPeriod"].notna()
        & (sales_only["HoldingPeriod"] != "LONG TERM")
    ]

    if not non_long_term.empty:
        logger.warning(
            "%d lot row(s) have a HoldingPeriod other than 'LONG TERM': %s",
            len(non_long_term),
            non_long_term["HoldingPeriod"]
            .value_counts()
            .to_dict(),
        )

        logger.warning(
            "Affected rows:\n%s",
            non_long_term[
                [
                    "Date",
                    "Symbol",
                    "GrantId",
                    "HoldingPeriod",
                ]
            ].to_string(index=False),
        )


# ============================================================================
# Vest command
# ============================================================================

@app.command("vest")
def compute_tax_on_rsu_vest(
    transactions_path: Path = typer.Option(..., exists=True),

    withholding_rate: float = typer.Option(
        0.22,
        help=(
            "Federal supplemental withholding rate used by Schwab "
            "at vest."
        ),
    ),

    target_rate: float = typer.Option(
        0.37,
        help=(
            "Fallback federal marginal rate used when "
            "--progressive-tax is not specified."
        ),
    ),

    ca_withholding_rate: float = typer.Option(
        0.1023,
        help=(
            "California supplemental withholding rate used by Schwab "
            "at vest."
        ),
    ),

    ca_target_rate: float = typer.Option(
        0.133,
        help=(
            "Fallback California marginal rate used when "
            "--progressive-tax is not specified."
        ),
    ),

    other_income: float = typer.Option(
        0.0,
        "--other-income",
        help=(
            "Expected full-year salary/ordinary income for the calendar "
            "year. Used by --progressive-tax."
        ),
    ),

    progressive_tax: bool = typer.Option(
        False,
        "--progressive-tax",
        help=(
            "Calculate incremental federal and California tax using "
            "progressive MFJ tax brackets and --other-income."
        ),
    ),

    start_date: Optional[datetime] = typer.Option(
        None,
        formats=["%Y-%m-%d"],
    ),

    end_date: Optional[datetime] = typer.Option(
        None,
        formats=["%Y-%m-%d"],
    ),
):
    df = pd.read_csv(transactions_path)

    df["TxnGroup"] = df["Date"].notna().cumsum()

    money_cols = [
        "Amount",
        "FairMarketValuePrice",
        "Taxes",
    ]

    for col in money_cols:
        if col in df.columns:
            df[col] = _clean_money(df[col])

    header_cols = [
        "Date",
        "Action",
        "Symbol",
        "Quantity",
    ]

    df[header_cols] = (
        df.groupby("TxnGroup")[header_cols]
        .ffill()
    )

    df = _filter_date_range(
        df,
        start_date,
        end_date,
    )

    lapses_only = df[df["Action"] == "Lapse"].copy()

    # ------------------------------------------------------------------------
    # Compute the actual vest amount from the lot rows.
    # ------------------------------------------------------------------------

    lot_shares = (
        lapses_only["SharesSoldWithheldForTaxes"]
        + lapses_only["NetSharesDeposited"]
    )

    lapses_only["LotShares"] = lot_shares

    lapses_only["LotVestAmount"] = (
        lot_shares
        * lapses_only["FairMarketValuePrice"]
    )

    summary = (
        lapses_only.groupby("TxnGroup")
        .agg(
            Date=("Date", "first"),
            Symbol=("Symbol", "first"),
            SharesVested=("Quantity", "first"),
            LotSharesSum=("LotShares", "sum"),
            FairMarketValuePrice=(
                "FairMarketValuePrice",
                "first",
            ),
            TotalVestAmount=("LotVestAmount", "sum"),
            TaxesWithheld=("Taxes", "sum"),
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------------
    # Handle stock-split discrepancies.
    # ------------------------------------------------------------------------

    ratio = (
        summary["SharesVested"]
        / summary["LotSharesSum"]
    )

    mismatched = summary[
        (ratio - 1).abs() > 0.01
    ]

    if not mismatched.empty:
        logger.warning(
            "%d vest group(s) have header Quantity != sum of lot shares "
            "(likely a stock split between vest and export date). "
            "Correcting TotalVestAmount using header Quantity:\n%s",
            len(mismatched),
            mismatched[
                [
                    "Date",
                    "SharesVested",
                    "LotSharesSum",
                    "TotalVestAmount",
                ]
            ].to_string(index=False),
        )

        summary.loc[
            mismatched.index,
            "TotalVestAmount",
        ] *= ratio[mismatched.index]

    # ------------------------------------------------------------------------
    # Validate options.
    # ------------------------------------------------------------------------

    if progressive_tax and other_income < 0:
        raise typer.BadParameter(
            "--other-income must be >= 0."
        )

    # ------------------------------------------------------------------------
    # Progressive calculation
    #
    # IMPORTANT:
    #
    # We calculate annual tax liability, not "tax based on salary received
    # so far."
    #
    # For example:
    #
    #   salary = $301,000
    #   March RSUs = $343,483
    #   June RSUs  = $405,071
    #
    # March incremental tax:
    #
    #   tax(salary + March RSUs)
    #   - tax(salary)
    #
    # June incremental tax:
    #
    #   tax(salary + March RSUs + June RSUs)
    #   - tax(salary + March RSUs)
    #
    # Thus the total exactly equals:
    #
    #   tax(salary + all RSUs)
    #   - tax(salary)
    #
    # The chronological allocation matters because March RSUs may push
    # income into higher brackets, causing June RSUs to have a higher
    # marginal tax rate.
    # ------------------------------------------------------------------------

    if progressive_tax:
        logger.info(
            "Using progressive MFJ tax calculation with "
            "full-year other income of $%,.2f.",
            other_income,
        )

        # Work at the vest-date level first.
        #
        # Multiple transaction groups can vest on the same date. They should
        # collectively occupy the same place in the progressive calculation.
        by_date = (
            summary.groupby("Date", as_index=False)
            .agg(
                TotalVestAmount=("TotalVestAmount", "sum"),
            )
        )

        by_date["DateParsed"] = pd.to_datetime(
            by_date["Date"]
        )

        by_date = (
            by_date
            .sort_values("DateParsed")
            .reset_index(drop=True)
        )

        previous_rsut_income = 0.0

        progressive_date_results = []

        base_federal_tax = _federal_income_tax(
            other_income
        )

        base_ca_tax = _california_income_tax(
            other_income
        )

        for _, row in by_date.iterrows():
            current_rsut_income = (
                previous_rsut_income
                + row["TotalVestAmount"]
            )

            federal_tax_before = _federal_income_tax(
                other_income + previous_rsut_income
            )

            federal_tax_after = _federal_income_tax(
                other_income + current_rsut_income
            )

            ca_tax_before = _california_income_tax(
                other_income + previous_rsut_income
            )

            ca_tax_after = _california_income_tax(
                other_income + current_rsut_income
            )

            incremental_federal_tax = (
                federal_tax_after
                - federal_tax_before
            )

            incremental_ca_tax = (
                ca_tax_after
                - ca_tax_before
            )

            vest_amount = float(
                row["TotalVestAmount"]
            )

            # Schwab's RSU withholding is represented by the configured
            # supplemental rates. The CSV "TaxesWithheld" field is a combined
            # amount and does not cleanly identify federal vs CA withholding,
            # so use the known rates rather than subtracting the entire field
            # from either tax jurisdiction.
            federal_withholding = (
                vest_amount
                * withholding_rate
            )

            ca_withholding = (
                vest_amount
                * ca_withholding_rate
            )

            additional_federal_tax = (
                incremental_federal_tax
                - federal_withholding
            )

            additional_ca_tax = (
                incremental_ca_tax
                - ca_withholding
            )

            progressive_date_results.append(
                {
                    "Date": row["Date"],
                    "VestAmount": vest_amount,
                    "FederalTaxAttributableToVest": (
                        incremental_federal_tax
                    ),
                    "FederalWithholding": (
                        federal_withholding
                    ),
                    "AdditionalFederalTaxDue": (
                        additional_federal_tax
                    ),
                    "CATaxAttributableToVest": (
                        incremental_ca_tax
                    ),
                    "CAWithholding": (
                        ca_withholding
                    ),
                    "AdditionalCATaxDue": (
                        additional_ca_tax
                    ),
                }
            )

            previous_rsut_income = current_rsut_income

        progressive_by_date = pd.DataFrame(
            progressive_date_results
        )

        # --------------------------------------------------------------------
        # Allocate each date's progressive tax back to the individual
        # transaction groups on that date, proportional to vest amount.
        # --------------------------------------------------------------------

        summary["Date"] = summary["Date"].astype(str)

        summary = summary.merge(
            progressive_by_date,
            on="Date",
            how="left",
        )

        summary["VestAmountFractionOfDate"] = (
            summary["TotalVestAmount"]
            / summary["VestAmount"]
        )

        summary["EstimatedFederalTaxOwed"] = (
            summary["FederalTaxAttributableToVest"]
            * summary["VestAmountFractionOfDate"]
        )

        summary["FederalWithholding"] = (
            summary["FederalWithholding"]
            * summary["VestAmountFractionOfDate"]
        )

        summary["AdditionalFederalTaxDue"] = (
            summary["AdditionalFederalTaxDue"]
            * summary["VestAmountFractionOfDate"]
        )

        summary["EstimatedCATaxOwed"] = (
            summary["CATaxAttributableToVest"]
            * summary["VestAmountFractionOfDate"]
        )

        summary["CAWithholding"] = (
            summary["CAWithholding"]
            * summary["VestAmountFractionOfDate"]
        )

        summary["AdditionalCATaxDue"] = (
            summary["AdditionalCATaxDue"]
            * summary["VestAmountFractionOfDate"]
        )

        summary["AdditionalTotalTaxDue"] = (
            summary["AdditionalFederalTaxDue"]
            + summary["AdditionalCATaxDue"]
        )

        # Drop intermediate calculation columns from the detailed output.
        summary = summary.drop(
            columns=[
                "VestAmount",
                "FederalTaxAttributableToVest",
                "CATaxAttributableToVest",
                "VestAmountFractionOfDate",
            ]
        )

    else:
        # --------------------------------------------------------------------
        # Original/simple calculation.
        #
        # Federal:
        #
        #   37% tax
        #   - 22% withholding
        #   = 15% additional
        #
        # California:
        #
        #   13.3% tax
        #   - 10.23% withholding
        #   = 3.07% additional
        # --------------------------------------------------------------------

        additional_rate = (
            target_rate
            - withholding_rate
        )

        ca_additional_rate = (
            ca_target_rate
            - ca_withholding_rate
        )

        summary["EstimatedFederalTaxOwed"] = (
            summary["TotalVestAmount"]
            * target_rate
        ).round(2)

        summary["FederalWithholding"] = (
            summary["TotalVestAmount"]
            * withholding_rate
        ).round(2)

        summary["AdditionalFederalTaxDue"] = (
            summary["TotalVestAmount"]
            * additional_rate
        ).round(2)

        summary["EstimatedCATaxOwed"] = (
            summary["TotalVestAmount"]
            * ca_target_rate
        ).round(2)

        summary["CAWithholding"] = (
            summary["TotalVestAmount"]
            * ca_withholding_rate
        ).round(2)

        summary["AdditionalCATaxDue"] = (
            summary["TotalVestAmount"]
            * ca_additional_rate
        ).round(2)

        summary["AdditionalTotalTaxDue"] = (
            summary["AdditionalFederalTaxDue"]
            + summary["AdditionalCATaxDue"]
        )

    # ------------------------------------------------------------------------
    # Round detailed output.
    # ------------------------------------------------------------------------

    money_columns = [
        "TotalVestAmount",
        "TaxesWithheld",
        "EstimatedFederalTaxOwed",
        "FederalWithholding",
        "AdditionalFederalTaxDue",
        "EstimatedCATaxOwed",
        "CAWithholding",
        "AdditionalCATaxDue",
        "AdditionalTotalTaxDue",
    ]

    for col in money_columns:
        if col in summary.columns:
            summary[col] = summary[col].round(2)

    print(summary)

    # ------------------------------------------------------------------------
    # Per-date summary.
    # ------------------------------------------------------------------------

    by_date_agg = {
        "SharesVested": ("SharesVested", "sum"),
        "TotalVestAmount": ("TotalVestAmount", "sum"),
        "TaxesWithheld": ("TaxesWithheld", "sum"),
        "EstimatedFederalTaxOwed": (
            "EstimatedFederalTaxOwed",
            "sum",
        ),
        "FederalWithholding": (
            "FederalWithholding",
            "sum",
        ),
        "AdditionalFederalTaxDue": (
            "AdditionalFederalTaxDue",
            "sum",
        ),
        "EstimatedCATaxOwed": (
            "EstimatedCATaxOwed",
            "sum",
        ),
        "CAWithholding": (
            "CAWithholding",
            "sum",
        ),
        "AdditionalCATaxDue": (
            "AdditionalCATaxDue",
            "sum",
        ),
        "AdditionalTotalTaxDue": (
            "AdditionalTotalTaxDue",
            "sum",
        ),
    }

    by_date_summary = (
        summary.groupby("Date")
        .agg(**by_date_agg)
        .reset_index()
    )

    by_date_summary["DateParsed"] = pd.to_datetime(
        by_date_summary["Date"]
    )

    by_date_summary = (
        by_date_summary
        .sort_values("DateParsed")
        .drop(columns=["DateParsed"])
        .reset_index(drop=True)
    )

    total_fed_due = (
        by_date_summary["AdditionalFederalTaxDue"]
        .sum()
    )

    total_ca_due = (
        by_date_summary["AdditionalCATaxDue"]
        .sum()
    )

    total_additional_due = (
        by_date_summary["AdditionalTotalTaxDue"]
        .sum()
    )

    by_date_summary = _append_total_row(
        by_date_summary
    )

    print("\nPer vest date:")
    print(by_date_summary)

    # ------------------------------------------------------------------------
    # Final summary.
    # ------------------------------------------------------------------------

    if progressive_tax:
        total_vest_amount = summary["TotalVestAmount"].sum()

        gross_federal_tax = (
            summary["EstimatedFederalTaxOwed"].sum()
        )

        gross_ca_tax = (
            summary["EstimatedCATaxOwed"].sum()
        )

        federal_withholding = (
            summary["FederalWithholding"].sum()
        )

        ca_withholding = (
            summary["CAWithholding"].sum()
        )

        print(
            "\nProgressive tax mode:"
        )

        print(
            f"  Filing status: Married Filing Jointly"
        )

        print(
            f"  Full-year other income: "
            f"${other_income:,.2f}"
        )

        print(
            f"  RSU income: "
            f"${total_vest_amount:,.2f}"
        )

        print(
            f"  Federal tax attributable to RSUs: "
            f"${gross_federal_tax:,.2f}"
        )

        print(
            f"  Federal RSU withholding: "
            f"${federal_withholding:,.2f}"
        )

        print(
            f"  Additional federal tax due: "
            f"${total_fed_due:,.2f}"
        )

        print(
            f"  CA tax attributable to RSUs: "
            f"${gross_ca_tax:,.2f}"
        )

        print(
            f"  CA RSU withholding: "
            f"${ca_withholding:,.2f}"
        )

        print(
            f"  Additional CA tax due: "
            f"${total_ca_due:,.2f}"
        )

        print(
            f"  TOTAL ADDITIONAL TAX DUE: "
            f"${total_additional_due:,.2f}"
        )

    else:
        print(
            f"\nAdditional federal prepayment needed "
            f"(~{(target_rate - withholding_rate) * 100:.1f}% "
            "of vest amount): "
            f"${total_fed_due:,.2f}"
        )

        print(
            f"Additional CA prepayment needed "
            f"(~{(ca_target_rate - ca_withholding_rate) * 100:.1f}% "
            "of vest amount): "
            f"${total_ca_due:,.2f}"
        )

        print(
            "Total additional prepayment needed across all vests: "
            f"${total_additional_due:,.2f}"
        )


if __name__ == "__main__":
    app()

