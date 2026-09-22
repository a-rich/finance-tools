import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pdfplumber
import typer

app = typer.Typer(no_args_is_help=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("pdfminer.pdffont").setLevel(logging.ERROR)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


@dataclass(frozen=True)
class PaySummary:
    tax_year: int
    report_date: datetime
    salary: float
    restricted_stock: float
    employer_hsa: float
    group_term_life: float
    employee_401k: float
    dental: float
    federal_income_tax_withheld: float
    ca_income_tax_withheld: float
    medicare_tax_withheld: float


@dataclass(frozen=True)
class YearEndTaxEstimate:
    federal_income: float
    ca_income: float
    federal_income_tax: float
    ca_income_tax: float
    additional_medicare_tax: float
    additional_medicare_withheld: float
    federal_balance: float
    ca_balance: float
    federal_prepayment: float
    ca_prepayment: float


def _money_from_match(match: re.Match[str] | None, field: str) -> float:
    if match is None:
        raise ValueError(f"Could not find {field} in the Workday pay summary.")

    return float(match.group("amount").replace(",", ""))


def _earnings_rows(text: str) -> list[tuple[str, float]]:
    """Reassemble Workday earnings labels that wrap below their values."""
    rows: list[tuple[str, float]] = []
    row_pattern = re.compile(
        r"^(?P<label>.*?)\s+[\d,.]+\s+\$(?P<amount>[\d,]+\.\d{2})$"
    )

    for line in text.splitlines():
        line = line.strip()
        match = row_pattern.match(line)
        if match is not None:
            rows.append(
                (
                    match.group("label"),
                    float(match.group("amount").replace(",", "")),
                )
            )
        elif rows and line and rows[-1][0].casefold() != "total":
            label, amount = rows[-1]
            rows[-1] = (f"{label} {line}", amount)

    return rows


def parse_pay_summary_text(
    text: str,
    report_date: datetime,
) -> PaySummary:
    """Parse the year-end values used by the vest calculation."""
    year_match = re.search(r"\bYear\s+(?P<year>20\d{2})\b", text)
    if year_match is None:
        raise ValueError("Could not find tax year in the Workday pay summary.")

    def section(name: str) -> str:
        match = re.search(
            rf"\[{name}\](?P<body>.*?)(?=\n\[[A-Z]+\]|\Z)",
            text,
            flags=re.DOTALL,
        )
        if match is None:
            raise ValueError(
                f"Could not find {name.lower()} section in the "
                "Workday pay summary."
            )
        return match.group("body")

    earnings = section("EARNINGS")
    deductions = section("DEDUCTIONS")
    taxes = section("TAXES")
    earnings_rows = _earnings_rows(earnings)

    def earnings_amount(label: str) -> float:
        for row_label, amount in earnings_rows:
            if re.fullmatch(label, row_label, flags=re.IGNORECASE):
                return amount
        raise ValueError(f"Could not find {label} in the Workday pay summary.")

    def first_money_after(source: str, label: str) -> float:
        return _money_from_match(
            re.search(
                rf"{label}\s+\$(?P<amount>[\d,]+\.\d{{2}})",
                source,
                flags=re.IGNORECASE,
            ),
            label,
        )

    salary = earnings_amount("Salary")
    restricted_stock = earnings_amount(r"Restricted\s+Stoc(?:k)?")
    employer_hsa = earnings_amount(r"Employer\s+HSA\s+Co")
    group_term_life = earnings_amount(r"Group\s+Term\s+Life")
    reported_earnings_total = earnings_amount("Total")
    recognized_earnings_total = (
        salary + restricted_stock + employer_hsa + group_term_life
    )
    if not math.isclose(
        reported_earnings_total,
        recognized_earnings_total,
        rel_tol=0.0,
        abs_tol=0.01,
    ):
        raise ValueError(
            "The Workday pay summary contains unsupported earnings: "
            f"reported total ${reported_earnings_total:,.2f}, recognized "
            f"total ${recognized_earnings_total:,.2f}."
        )

    employee_401k = first_money_after(deductions, r"401\(k\)")
    dental = first_money_after(deductions, "Dental")
    rsu_deductions = []
    for amount_text in re.findall(
        r"\bRSU\s+(\(?\$[\d,]+\.\d{2}\)?)",
        deductions,
        flags=re.IGNORECASE,
    ):
        negative = amount_text.startswith("(")
        amount = float(amount_text.strip("()$").replace(",", ""))
        rsu_deductions.append(-amount if negative else amount)

    reported_deductions_total = _money_from_match(
        re.search(
            r"\bTotal\s+\$(?P<amount>[\d,]+\.\d{2})",
            deductions,
            flags=re.IGNORECASE,
        ),
        "total employee deductions",
    )
    recognized_deductions_total = (
        employee_401k
        + dental
        + employer_hsa
        + group_term_life
        + sum(rsu_deductions)
    )
    if not math.isclose(
        reported_deductions_total,
        recognized_deductions_total,
        rel_tol=0.0,
        abs_tol=0.01,
    ):
        raise ValueError(
            "The Workday pay summary contains unsupported deductions: "
            f"reported employee total ${reported_deductions_total:,.2f}, "
            f"recognized total ${recognized_deductions_total:,.2f}."
        )

    return PaySummary(
        tax_year=int(year_match.group("year")),
        report_date=report_date,
        salary=salary,
        restricted_stock=restricted_stock,
        employer_hsa=employer_hsa,
        group_term_life=group_term_life,
        employee_401k=employee_401k,
        dental=dental,
        federal_income_tax_withheld=first_money_after(
            taxes,
            "Federal",
        ),
        ca_income_tax_withheld=first_money_after(
            taxes,
            r"CA\s+State",
        ),
        medicare_tax_withheld=first_money_after(
            taxes,
            "Employee",
        ),
    )


def load_pay_summary(path: Path) -> PaySummary:
    """Extract a Workday pay summary from its three-column PDF layout."""
    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) != 1:
            raise ValueError(
                "Expected the Workday pay summary PDF to contain one page."
            )

        page = pdf.pages[0]
        full_text = page.extract_text() or ""
        report_date_match = re.search(
            r"(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),",
            full_text,
        )
        if report_date_match is None:
            raise ValueError(
                "Could not find the report date in the Workday pay summary."
            )

        date_text = report_date_match.group("date")
        date_format = (
            "%m/%d/%Y" if len(date_text.rsplit("/", 1)[1]) == 4 else "%m/%d/%y"
        )
        report_date = datetime.strptime(date_text, date_format)

        words = page.extract_words()

        def column_left(name: str) -> float:
            headings = [
                word
                for word in words
                if word["text"].casefold() == name.casefold()
            ]
            if not headings:
                raise ValueError(
                    f"Could not find {name.lower()} column in the "
                    "Workday pay summary."
                )

            heading = min(headings, key=lambda word: word["top"])
            horizontal_tolerance = page.width * 0.03
            candidates = [
                word["x0"]
                for word in words
                if heading["top"] <= word["top"] <= heading["bottom"] + 50
                and heading["x0"] - horizontal_tolerance
                <= word["x0"]
                <= heading["x0"] + horizontal_tolerance
            ]
            return max(0, min(candidates, default=heading["x0"]) - 1)

        deductions_left = column_left("Deductions")
        taxes_left = column_left("Taxes")
        earnings = (
            page.crop((0, 0, deductions_left, page.height)).extract_text()
            or ""
        )
        deductions = (
            page.crop(
                (deductions_left, 0, taxes_left, page.height)
            ).extract_text()
            or ""
        )
        taxes = (
            page.crop((taxes_left, 0, page.width, page.height)).extract_text()
            or ""
        )

    sectioned_text = (
        f"{full_text}\n"
        f"[EARNINGS]\n{earnings}\n"
        f"[DEDUCTIONS]\n{deductions}\n"
        f"[TAXES]\n{taxes}"
    )
    return parse_pay_summary_text(
        sectioned_text,
        report_date=report_date,
    )


# ============================================================================
# Tax constants
# ============================================================================

# Federal tax brackets for Married Filing Jointly.
#
# Each tuple is:
#     (upper_bound, marginal_rate)
#
# None means there is no upper bound.
#
# Sources: IRS Revenue Procedure 2024-40, the 2025 Working Families Tax
# Cuts adjustments, and IRS 2026 tax rate tables.
IRS_MFJ_BRACKETS_BY_YEAR = {
    2025: [
        (23_850, 0.10),
        (96_950, 0.12),
        (206_700, 0.22),
        (394_600, 0.24),
        (501_050, 0.32),
        (751_600, 0.35),
        (None, 0.37),
    ],
    2026: [
        (24_800, 0.10),
        (100_800, 0.12),
        (211_400, 0.22),
        (403_550, 0.24),
        (512_450, 0.32),
        (768_700, 0.35),
        (None, 0.37),
    ],
}

IRS_MFJ_STANDARD_DEDUCTION_BY_YEAR = {
    2025: 31_500,
    2026: 32_200,
}


# California's 2026 estimated-tax instructions currently direct taxpayers
# to use the 2025 tax table. These are therefore the latest published
# California Schedule Y brackets as of this script's 2026 tax-year use.
#
# Source: California FTB 2025 Schedule Y.
CA_MFJ_BRACKETS_2025 = [
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

CA_MFJ_STANDARD_DEDUCTION_2025 = 11_412

CA_MFJ_BRACKETS_BY_YEAR = {
    2025: CA_MFJ_BRACKETS_2025,
    2026: CA_MFJ_BRACKETS_2025,
}

CA_MFJ_STANDARD_DEDUCTION_BY_YEAR = {
    2025: CA_MFJ_STANDARD_DEDUCTION_2025,
    2026: CA_MFJ_STANDARD_DEDUCTION_2025,
}

# California Behavioral Health Services Tax:
# additional 1% on taxable income above $1,000,000.
CA_BEHAVIORAL_HEALTH_THRESHOLD = 1_000_000
CA_BEHAVIORAL_HEALTH_RATE = 0.01

MEDICARE_RATE = 0.0145
ADDITIONAL_MEDICARE_RATE = 0.009
ADDITIONAL_MEDICARE_MFJ_THRESHOLD = 250_000


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


def load_vest_summary(transactions_path: Path) -> pd.DataFrame:
    """Load Schwab vest transactions and summarize each transaction group."""
    df = pd.read_csv(transactions_path)
    df["TxnGroup"] = df["Date"].notna().cumsum()

    for col in ["Amount", "FairMarketValuePrice", "Taxes"]:
        if col in df.columns:
            df[col] = _clean_money(df[col])

    header_cols = ["Date", "Action", "Symbol", "Quantity"]
    df[header_cols] = df.groupby("TxnGroup")[header_cols].ffill()
    lapses_only = df[df["Action"] == "Lapse"].copy()
    lapses_only["LotShares"] = (
        lapses_only["SharesSoldWithheldForTaxes"]
        + lapses_only["NetSharesDeposited"]
    )
    lapses_only["LotVestAmount"] = (
        lapses_only["LotShares"] * lapses_only["FairMarketValuePrice"]
    )

    summary = (
        lapses_only.groupby("TxnGroup")
        .agg(
            Date=("Date", "first"),
            Symbol=("Symbol", "first"),
            SharesVested=("Quantity", "first"),
            LotSharesSum=("LotShares", "sum"),
            FairMarketValuePrice=("FairMarketValuePrice", "first"),
            TotalVestAmount=("LotVestAmount", "sum"),
            TaxesWithheld=("Taxes", "sum"),
        )
        .reset_index(drop=True)
    )

    ratio = summary["SharesVested"] / summary["LotSharesSum"]
    mismatched = summary[(ratio - 1).abs() > 0.01]
    if not mismatched.empty:
        details = mismatched[
            ["Date", "SharesVested", "LotSharesSum"]
        ].to_string(index=False)
        raise ValueError(
            "A Schwab vest header and its lot share total do not match. "
            "The export may be incomplete; no automatic stock-split "
            f"correction was applied:\n{details}"
        )

    return summary


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

        amount_in_bracket = min(taxable_income, upper_bound) - lower_bound

        tax += amount_in_bracket * rate
        lower_bound = upper_bound

    return tax


def _federal_income_tax(gross_income: float, tax_year: int) -> float:
    """Approximate federal ordinary income tax for MFJ.

    Uses the selected year's standard deduction and MFJ brackets.
    """
    taxable_income = max(
        0.0,
        gross_income - IRS_MFJ_STANDARD_DEDUCTION_BY_YEAR[tax_year],
    )

    return _progressive_tax(
        taxable_income,
        IRS_MFJ_BRACKETS_BY_YEAR[tax_year],
    )


def _california_income_tax(gross_income: float, tax_year: int) -> float:
    """Approximate California income tax for MFJ.

    Uses the applicable California Schedule Y brackets and MFJ standard
    deduction.

    Also includes the 1% Behavioral Health Services Tax on taxable
    income above $1,000,000.
    """
    taxable_income = max(
        0.0,
        gross_income - CA_MFJ_STANDARD_DEDUCTION_BY_YEAR[tax_year],
    )

    regular_tax = _progressive_tax(
        taxable_income,
        CA_MFJ_BRACKETS_BY_YEAR[tax_year],
    )

    behavioral_health_tax = (
        max(
            0.0,
            taxable_income - CA_BEHAVIORAL_HEALTH_THRESHOLD,
        )
        * CA_BEHAVIORAL_HEALTH_RATE
    )

    return regular_tax + behavioral_health_tax


def estimate_year_end_tax(
    pay_summary: PaySummary,
    total_income: Optional[float] = None,
) -> YearEndTaxEstimate:
    """Estimate year-end tax and remaining payments from Workday totals."""
    if (
        pay_summary.tax_year not in IRS_MFJ_BRACKETS_BY_YEAR
        or pay_summary.tax_year not in CA_MFJ_BRACKETS_BY_YEAR
    ):
        raise ValueError(f"Unsupported tax year: {pay_summary.tax_year}.")

    if total_income is not None and (
        not math.isfinite(total_income) or total_income < 0
    ):
        raise ValueError("Total income must be a finite, nonnegative number.")

    federal_income = (
        pay_summary.salary
        + pay_summary.restricted_stock
        + pay_summary.group_term_life
        - pay_summary.employee_401k
        - pay_summary.dental
    )
    ca_income = (
        pay_summary.salary
        + pay_summary.restricted_stock
        + pay_summary.employer_hsa
        + pay_summary.group_term_life
        - pay_summary.employee_401k
        - pay_summary.dental
    )
    if total_income is not None:
        federal_income = total_income
        ca_income = total_income

    medicare_wages = (
        pay_summary.salary
        + pay_summary.restricted_stock
        + pay_summary.group_term_life
        - pay_summary.dental
    )

    federal_income_tax = _federal_income_tax(
        federal_income,
        pay_summary.tax_year,
    )
    ca_income_tax = _california_income_tax(
        ca_income,
        pay_summary.tax_year,
    )
    additional_medicare_tax = (
        max(
            0.0,
            medicare_wages - ADDITIONAL_MEDICARE_MFJ_THRESHOLD,
        )
        * ADDITIONAL_MEDICARE_RATE
    )
    additional_medicare_withheld = max(
        0.0,
        pay_summary.medicare_tax_withheld - medicare_wages * MEDICARE_RATE,
    )

    federal_balance = (
        federal_income_tax
        + additional_medicare_tax
        - pay_summary.federal_income_tax_withheld
        - additional_medicare_withheld
    )
    ca_balance = ca_income_tax - pay_summary.ca_income_tax_withheld

    return YearEndTaxEstimate(
        federal_income=federal_income,
        ca_income=ca_income,
        federal_income_tax=federal_income_tax,
        ca_income_tax=ca_income_tax,
        additional_medicare_tax=additional_medicare_tax,
        additional_medicare_withheld=additional_medicare_withheld,
        federal_balance=federal_balance,
        ca_balance=ca_balance,
        federal_prepayment=max(0.0, federal_balance),
        ca_prepayment=max(0.0, ca_balance),
    )


def validate_year_end_inputs(
    pay_summary: PaySummary,
    vest_total: float,
) -> None:
    """Reject inputs that cannot support a year-end tax estimate."""
    if not math.isclose(
        vest_total,
        pay_summary.restricted_stock,
        rel_tol=0.0,
        abs_tol=0.01,
    ):
        raise ValueError(
            "Workday restricted-stock income and the Schwab vest total do "
            f"not match: ${pay_summary.restricted_stock:,.2f} versus "
            f"${vest_total:,.2f}."
        )


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
        summary["EstimatedIRSTax"] + summary["EstimatedCATax"]
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
            non_long_term["HoldingPeriod"].value_counts().to_dict(),
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
    pay_summary_path: Path = typer.Option(..., exists=True),
    total_income: Optional[float] = typer.Option(
        None,
        help=(
            "Override the pre-deduction federal and California income base."
        ),
    ),
):
    """Estimate year-end tax payments from Schwab and Workday records."""
    try:
        pay_summary = load_pay_summary(pay_summary_path)
        vest_summary = load_vest_summary(transactions_path)
        vest_summary["DateParsed"] = pd.to_datetime(
            vest_summary["Date"],
            format="%m/%d/%Y",
        )
        year_vests = vest_summary[
            vest_summary["DateParsed"].dt.year == pay_summary.tax_year
        ].copy()
        if year_vests.empty:
            raise ValueError(
                "The Schwab export contains no vest transactions for "
                f"{pay_summary.tax_year}."
            )

        vest_total = float(year_vests["TotalVestAmount"].sum())
        validate_year_end_inputs(
            pay_summary,
            vest_total=vest_total,
        )
        estimate = estimate_year_end_tax(
            pay_summary,
            total_income=total_income,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    print(f"Year-end tax estimate ({pay_summary.tax_year})")
    print("  Filing status: Married Filing Jointly")
    print("  Deduction assumption: federal and California standard deductions")
    print(f"  Workday report date: {pay_summary.report_date:%Y-%m-%d}")
    if total_income is not None:
        print(f"  User-provided total income: ${total_income:,.2f}")
    print(f"  Salary: ${pay_summary.salary:,.2f}")
    print(f"  RSU income: ${pay_summary.restricted_stock:,.2f}")
    print(
        "  Pretax 401(k) and dental deductions: "
        f"${pay_summary.employee_401k + pay_summary.dental:,.2f}"
    )
    print(
        "  Schwab reconciliation: "
        f"{len(year_vests)} vest group(s), ${vest_total:,.2f}"
    )

    print("\nFederal:")
    print(
        f"  Estimated taxable compensation before standard deduction: "
        f"${estimate.federal_income:,.2f}"
    )
    print(
        f"  Estimated federal income tax: ${estimate.federal_income_tax:,.2f}"
    )
    print(
        f"  Federal income tax withheld: "
        f"${pay_summary.federal_income_tax_withheld:,.2f}"
    )
    print(
        f"  Additional Medicare Tax: ${estimate.additional_medicare_tax:,.2f}"
    )
    print(
        f"  Additional Medicare Tax withheld: "
        f"${estimate.additional_medicare_withheld:,.2f}"
    )
    print(f"  Estimated federal balance: ${estimate.federal_balance:,.2f}")
    print(f"  Recommended IRS prepayment: ${estimate.federal_prepayment:,.2f}")

    print("\nCalifornia:")
    print(
        f"  Estimated California income before standard deduction: "
        f"${estimate.ca_income:,.2f}"
    )
    print(f"  Estimated California income tax: ${estimate.ca_income_tax:,.2f}")
    print(
        f"  California income tax withheld: "
        f"${pay_summary.ca_income_tax_withheld:,.2f}"
    )
    print(f"  Estimated California balance: ${estimate.ca_balance:,.2f}")
    print(f"  Recommended CA FTB prepayment: ${estimate.ca_prepayment:,.2f}")

    if total_income is None:
        print(
            "\nThis employer-only estimate excludes spouse and outside income, "
            "credits, prior estimated payments, and sale-related tax."
        )
    else:
        print(
            "\nThis estimate uses the user-provided total income for federal "
            "and California income tax. It excludes credits, prior estimated "
            "payments, and sale-related tax."
        )


if __name__ == "__main__":
    app()
