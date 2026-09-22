from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "calc_taxes.py"
SPEC = importlib.util.spec_from_file_location("calc_taxes", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
calc_taxes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calc_taxes)


def _write_text_pdf(
    path: Path,
    positioned_lines: list[tuple[int, int, str]],
) -> None:
    """Write a tiny real PDF for exercising the PDF extraction boundary."""
    commands = []
    for x, y, value in positioned_lines:
        escaped = (
            value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        )
        commands.append(f"BT /F1 9 Tf {x} {y} Td ({escaped}) Tj ET")
    stream = "\n".join(commands).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> "
            b"/Contents 4 0 R >>"
        ),
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(pdf)


def _write_pay_summary_pdf(
    path: Path,
    *,
    report_date: str = "12/31/26",
    year: int = 2026,
    salary: float = 300_000.00,
    restricted_stock: float = 1_100_000.00,
    federal_withholding: float = 310_000.00,
    ca_withholding: float = 130_000.00,
    medicare_withholding: float = 30_000.00,
    deductions_x: int = 270,
    taxes_x: int = 450,
) -> None:
    employer_hsa = 2_500.00
    group_term_life = 500.00
    employee_401k = 23_000.00
    dental = 500.00
    total_earnings = salary + restricted_stock + employer_hsa + group_term_life
    total_deductions = employee_401k + dental + employer_hsa + group_term_life
    _write_text_pdf(
        path,
        [
            (34, 760, f"{report_date}, 6:03 PM"),
            (34, 700, f"Year {year}"),
            (34, 615, "Earnings"),
            (34, 595, f"Salary 2,080.0000 ${salary:,.2f}"),
            (
                34,
                580,
                f"Restricted Stoc 0.0000 ${restricted_stock:,.2f}",
            ),
            (34, 565, f"Employer HSA Co 0.0000 ${employer_hsa:,.2f}"),
            (34, 550, f"Group Term Life 0.0000 ${group_term_life:,.2f}"),
            (34, 535, f"Total 2,080.0000 ${total_earnings:,.2f}"),
            (deductions_x, 615, "Deductions"),
            (
                deductions_x,
                595,
                f"401(k) ${employee_401k:,.2f} $0.00",
            ),
            (deductions_x, 580, "RSU ($0.00) $0.00"),
            (deductions_x, 565, "RSU $0.00 $0.00"),
            (deductions_x, 550, f"Dental ${dental:,.2f} $0.00"),
            (
                deductions_x,
                535,
                f"Employer ${employer_hsa:,.2f} $0.00",
            ),
            (deductions_x, 520, f"Group ${group_term_life:,.2f} $0.00"),
            (
                deductions_x,
                505,
                f"Total ${total_deductions:,.2f} $0.00",
            ),
            (taxes_x, 615, "Taxes"),
            (taxes_x, 595, f"CA State ${ca_withholding:,.2f}"),
            (taxes_x, 580, f"Federal ${federal_withholding:,.2f}"),
            (taxes_x, 565, f"Employee ${medicare_withholding:,.2f}"),
        ],
    )


def test_sale_calculation_remains_unchanged(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catch changes to the existing sale tax and aggregation behavior."""
    transactions_path = tmp_path / "transactions.csv"
    pd.DataFrame(
        [
            {
                "Date": "09/04/2026",
                "Action": "Sale",
                "Symbol": "NVDA",
                "Quantity": 10.0,
                "Amount": "$2,000.00",
                "SalePrice": "$200.00",
                "FeesAndCommissions": "$1.00",
                "VestFairMarketValue": "$100.00",
                "TotalCostBasis": None,
                "RealizedGainLoss": None,
                "HoldingPeriod": None,
                "GrantId": None,
            },
            {
                "Date": None,
                "Action": None,
                "Symbol": None,
                "Quantity": None,
                "Amount": None,
                "SalePrice": None,
                "FeesAndCommissions": None,
                "VestFairMarketValue": None,
                "TotalCostBasis": "$1,000.00",
                "RealizedGainLoss": "$1,000.00",
                "HoldingPeriod": "LONG TERM",
                "GrantId": "grant-1",
            },
        ]
    ).to_csv(transactions_path, index=False)

    calc_taxes.compute_tax_on_rsu_sale(
        transactions_path=transactions_path,
        irs_rate=0.20,
        ca_rate=0.133,
        start_date=None,
        end_date=None,
    )

    output = capsys.readouterr().out
    assert "TotalRealizedGainLoss" in output
    assert "EstimatedIRSTax" in output
    assert "EstimatedCATax" in output
    assert "1000.0" in output
    assert "200.0" in output
    assert "133.0" in output


def test_parse_pay_summary_extracts_year_end_totals() -> None:
    """Catch missing or misclassified Workday year-end pay fields."""
    text = """
    Year 2026
    [EARNINGS]
    Earnings
    Type Hours Amount
    Salary 2,080.0000 $300,000.00
    Restricted Stoc 0.0000 $1,100,000.00
    Employer HSA Co 0.0000 $2,500.00
    Group Term Life 0.0000 $500.00
    Total 2,080.0000 $1,403,000.00
    [DEDUCTIONS]
    Deductions
    Employee Employer
    401(k) $23,000.00 $0.00
    plan
    RSU ($0.00) $0.00
    RSU $0.00 $0.00
    Dental $500.00 $0.00
    Employer $2,500.00 $0.00
    HSA Co
    Group $500.00 $0.00
    Term Life
    Total $26,500.00 $0.00
    [TAXES]
    Taxes
    Type Amount
    CA State $130,000.00
    Income
    Tax
    Federal $310,000.00
    Income
    Tax
    Employee $30,000.00
    Medicare
    """

    summary = calc_taxes.parse_pay_summary_text(
        text,
        report_date=calc_taxes.datetime(2026, 12, 31),
    )

    assert summary.tax_year == 2026
    assert summary.report_date == calc_taxes.datetime(2026, 12, 31)
    assert summary.salary == 300_000.00
    assert summary.restricted_stock == 1_100_000.00
    assert summary.employer_hsa == 2_500.00
    assert summary.group_term_life == 500.00
    assert summary.employee_401k == 23_000.00
    assert summary.dental == 500.00
    assert summary.federal_income_tax_withheld == 310_000.00
    assert summary.ca_income_tax_withheld == 130_000.00
    assert summary.medicare_tax_withheld == 30_000.00


def test_parse_pay_summary_reassembles_wrapped_earnings_labels() -> None:
    """Catch requiring a Workday earnings label to stay on one line."""
    text = """
    Year 2025
    [EARNINGS]
    Salary 2,080.0800 $120,000.00
    Restricted 0.0000 $500,000.00
    Stoc
    Employer 0.0000 $2,000.00
    HSA Co
    Group 0.0000 $300.00
    Term Life
    Total 2,080.0800 $622,300.00
    [DEDUCTIONS]
    401(k) $18,000.00 $0.00
    RSU ($1,000.00) $0.00
    RSU $500,000.00 $0.00
    Dental $300.00 $0.00
    Total $519,600.00 $0.00
    [TAXES]
    CA State $20,000.00
    Federal $50,000.00
    Employee $10,000.00
    """

    summary = calc_taxes.parse_pay_summary_text(
        text,
        report_date=calc_taxes.datetime(2026, 9, 21),
    )

    assert summary.restricted_stock == 500_000.00
    assert summary.employer_hsa == 2_000.00
    assert summary.group_term_life == 300.00


def test_parse_pay_summary_rejects_unclassified_earnings() -> None:
    """Catch silently omitting a nonzero Workday earnings row."""
    text = """
    Year 2026
    [EARNINGS]
    Salary 2,080.0000 $300,000.00
    Restricted Stoc 0.0000 $1,100,000.00
    Employer HSA Co 0.0000 $2,500.00
    Group Term Life 0.0000 $500.00
    Bonus 0.0000 $10,000.00
    Total 2,080.0000 $1,413,000.00
    [DEDUCTIONS]
    401(k) $23,000.00 $0.00
    Dental $500.00 $0.00
    [TAXES]
    CA State $130,000.00
    Federal $310,000.00
    Employee $30,000.00
    """

    with pytest.raises(ValueError, match="unsupported earnings"):
        calc_taxes.parse_pay_summary_text(
            text,
            report_date=calc_taxes.datetime(2026, 12, 31),
        )


def test_parse_pay_summary_rejects_unclassified_deductions() -> None:
    """Catch silently omitting a nonzero Workday deduction row."""
    text = """
    Year 2026
    [EARNINGS]
    Salary 2,080.0000 $300,000.00
    Restricted Stoc 0.0000 $1,100,000.00
    Employer HSA Co 0.0000 $2,500.00
    Group Term Life 0.0000 $500.00
    Total 2,080.0000 $1,403,000.00
    [DEDUCTIONS]
    401(k) $23,000.00 $0.00
    RSU ($0.00) $0.00
    Excess
    Tax
    RSU $0.00 $0.00
    Offset
    Dental $500.00 $0.00
    Employer $2,500.00 $0.00
    HSA Co
    Group $500.00 $0.00
    Term Life
    Other Pretax $1,000.00 $0.00
    Total $27,500.00 $0.00
    [TAXES]
    CA State $130,000.00
    Federal $310,000.00
    Employee $30,000.00
    """

    with pytest.raises(ValueError, match="unsupported deductions"):
        calc_taxes.parse_pay_summary_text(
            text,
            report_date=calc_taxes.datetime(2026, 12, 31),
        )


def test_load_pay_summary_extracts_workday_columns_from_pdf(
    tmp_path: Path,
) -> None:
    """Catch treating Workday's three parallel columns as one text stream."""
    pdf_path = tmp_path / "pay-summary.pdf"
    _write_pay_summary_pdf(pdf_path)

    summary = calc_taxes.load_pay_summary(pdf_path)

    assert summary.report_date == calc_taxes.datetime(2026, 12, 31)
    assert summary.salary == 300_000.00
    assert summary.restricted_stock == 1_100_000.00
    assert summary.employee_401k == 23_000.00
    assert summary.federal_income_tax_withheld == 310_000.00
    assert summary.ca_income_tax_withheld == 130_000.00


def test_load_pay_summary_detects_shifted_workday_columns(
    tmp_path: Path,
) -> None:
    """Catch clipping fields when Workday shifts its three columns."""
    pdf_path = tmp_path / "shifted-pay-summary.pdf"
    _write_pay_summary_pdf(
        pdf_path,
        deductions_x=244,
        taxes_x=443,
    )

    summary = calc_taxes.load_pay_summary(pdf_path)

    assert summary.employee_401k == 23_000.00
    assert summary.federal_income_tax_withheld == 310_000.00
    assert summary.ca_income_tax_withheld == 130_000.00


def test_estimate_year_end_tax_uses_actual_compensation_and_withholding() -> (
    None
):
    """Catch projecting income or ignoring actual year-end withholding."""
    pay_summary = calc_taxes.PaySummary(
        tax_year=2026,
        report_date=calc_taxes.datetime(2026, 12, 31),
        salary=300_000.00,
        restricted_stock=1_100_000.00,
        employer_hsa=2_500.00,
        group_term_life=500.00,
        employee_401k=23_000.00,
        dental=500.00,
        federal_income_tax_withheld=310_000.00,
        ca_income_tax_withheld=130_000.00,
        medicare_tax_withheld=30_000.00,
    )

    estimate = calc_taxes.estimate_year_end_tax(pay_summary)

    assert estimate.federal_income == 1_377_000.00
    assert estimate.ca_income == 1_379_500.00
    assert estimate.federal_income_tax == pytest.approx(419_740.50)
    assert estimate.additional_medicare_tax == pytest.approx(10_350.00)
    assert estimate.additional_medicare_withheld == pytest.approx(9_700.00)
    assert estimate.federal_balance == pytest.approx(110_390.50)
    assert estimate.federal_prepayment == pytest.approx(110_390.50)
    assert estimate.ca_income_tax == pytest.approx(134_807.10)
    assert estimate.ca_balance == pytest.approx(4_807.10)
    assert estimate.ca_prepayment == pytest.approx(4_807.10)


def test_estimate_year_end_tax_uses_2025_federal_parameters() -> None:
    """Catch applying the current year's brackets to a historical summary."""
    pay_summary = calc_taxes.PaySummary(
        tax_year=2025,
        report_date=calc_taxes.datetime(2026, 9, 21),
        salary=300_000.00,
        restricted_stock=1_100_000.00,
        employer_hsa=2_500.00,
        group_term_life=500.00,
        employee_401k=23_000.00,
        dental=500.00,
        federal_income_tax_withheld=310_000.00,
        ca_income_tax_withheld=130_000.00,
        medicare_tax_withheld=30_000.00,
    )

    estimate = calc_taxes.estimate_year_end_tax(pay_summary)

    assert estimate.federal_income_tax == pytest.approx(421_897.50)


def test_estimate_year_end_tax_rejects_unsupported_tax_year() -> None:
    """Catch silently calculating a year with another year's parameters."""
    pay_summary = calc_taxes.PaySummary(
        tax_year=2024,
        report_date=calc_taxes.datetime(2025, 1, 1),
        salary=100_000.00,
        restricted_stock=0.00,
        employer_hsa=0.00,
        group_term_life=0.00,
        employee_401k=0.00,
        dental=0.00,
        federal_income_tax_withheld=0.00,
        ca_income_tax_withheld=0.00,
        medicare_tax_withheld=0.00,
    )

    with pytest.raises(ValueError, match="Unsupported tax year: 2024"):
        calc_taxes.estimate_year_end_tax(pay_summary)


def test_validate_year_end_inputs_accepts_pay_summary_from_any_date() -> None:
    """Catch requiring a December 31 Workday report."""
    pay_summary = calc_taxes.PaySummary(
        tax_year=2026,
        report_date=calc_taxes.datetime(2026, 12, 20),
        salary=200_000.00,
        restricted_stock=1_100_000.00,
        employer_hsa=2_500.00,
        group_term_life=500.00,
        employee_401k=20_000.00,
        dental=500.00,
        federal_income_tax_withheld=300_000.00,
        ca_income_tax_withheld=120_000.00,
        medicare_tax_withheld=28_000.00,
    )

    calc_taxes.validate_year_end_inputs(
        pay_summary,
        vest_total=1_100_000.00,
    )


def test_validate_year_end_inputs_accepts_historical_summary_printed_later() -> (
    None
):
    """Catch treating Workday's print timestamp as the selected tax year."""
    pay_summary = calc_taxes.PaySummary(
        tax_year=2025,
        report_date=calc_taxes.datetime(2026, 9, 21),
        salary=300_000.00,
        restricted_stock=1_100_000.00,
        employer_hsa=2_500.00,
        group_term_life=500.00,
        employee_401k=23_000.00,
        dental=500.00,
        federal_income_tax_withheld=310_000.00,
        ca_income_tax_withheld=130_000.00,
        medicare_tax_withheld=30_000.00,
    )

    calc_taxes.validate_year_end_inputs(
        pay_summary,
        vest_total=1_100_000.00,
    )


def test_validate_year_end_inputs_reconciles_rsu_income() -> None:
    """Catch calculating from an incomplete Schwab export."""
    pay_summary = calc_taxes.PaySummary(
        tax_year=2026,
        report_date=calc_taxes.datetime(2026, 12, 31),
        salary=300_000.00,
        restricted_stock=1_100_000.00,
        employer_hsa=2_500.00,
        group_term_life=500.00,
        employee_401k=23_000.00,
        dental=500.00,
        federal_income_tax_withheld=310_000.00,
        ca_income_tax_withheld=130_000.00,
        medicare_tax_withheld=30_000.00,
    )

    with pytest.raises(ValueError, match="do not match"):
        calc_taxes.validate_year_end_inputs(
            pay_summary,
            vest_total=1_000_000.00,
        )


def test_load_vest_summary_uses_schwab_lot_values(
    tmp_path: Path,
) -> None:
    """Catch trusting a header value instead of Schwab's vest lot values."""
    transactions_path = tmp_path / "transactions.csv"
    pd.DataFrame(
        [
            {
                "Date": "12/16/2026",
                "Action": "Lapse",
                "Symbol": "NVDA",
                "Quantity": 10.0,
                "FairMarketValuePrice": None,
                "SharesSoldWithheldForTaxes": None,
                "NetSharesDeposited": None,
                "Taxes": None,
            },
            {
                "Date": None,
                "Action": None,
                "Symbol": None,
                "Quantity": None,
                "FairMarketValuePrice": "$100.00",
                "SharesSoldWithheldForTaxes": 4.0,
                "NetSharesDeposited": 6.0,
                "Taxes": "$400.00",
            },
        ]
    ).to_csv(transactions_path, index=False)

    summary = calc_taxes.load_vest_summary(transactions_path)

    assert summary.loc[0, "Date"] == "12/16/2026"
    assert summary.loc[0, "SharesVested"] == 10.0
    assert summary.loc[0, "TotalVestAmount"] == 1_000.00
    assert summary.loc[0, "TaxesWithheld"] == 400.00


def test_load_vest_summary_rejects_missing_lot_shares(
    tmp_path: Path,
) -> None:
    """Catch scaling an incomplete Schwab lot export into a plausible total."""
    transactions_path = tmp_path / "transactions.csv"
    pd.DataFrame(
        [
            {
                "Date": "12/16/2026",
                "Action": "Lapse",
                "Symbol": "NVDA",
                "Quantity": 10.0,
                "FairMarketValuePrice": None,
                "SharesSoldWithheldForTaxes": None,
                "NetSharesDeposited": None,
                "Taxes": None,
            },
            {
                "Date": None,
                "Action": None,
                "Symbol": None,
                "Quantity": None,
                "FairMarketValuePrice": "$100.00",
                "SharesSoldWithheldForTaxes": 4.0,
                "NetSharesDeposited": 5.0,
                "Taxes": "$400.00",
            },
        ]
    ).to_csv(transactions_path, index=False)

    with pytest.raises(ValueError, match="share total"):
        calc_taxes.load_vest_summary(transactions_path)


def test_vest_cli_uses_one_pay_summary_as_final_year_amounts(
    tmp_path: Path,
) -> None:
    """Catch date gating, projections, manual modes, or a paycheck input."""
    transactions_path = tmp_path / "transactions.csv"
    pd.DataFrame(
        [
            {
                "Date": "12/16/2026",
                "Action": "Lapse",
                "Symbol": "NVDA",
                "Quantity": 10.0,
                "FairMarketValuePrice": None,
                "SharesSoldWithheldForTaxes": None,
                "NetSharesDeposited": None,
                "Taxes": None,
            },
            {
                "Date": None,
                "Action": None,
                "Symbol": None,
                "Quantity": None,
                "FairMarketValuePrice": "$100.00",
                "SharesSoldWithheldForTaxes": 4.0,
                "NetSharesDeposited": 6.0,
                "Taxes": "$400.00",
            },
        ]
    ).to_csv(transactions_path, index=False)

    pay_summary_path = tmp_path / "pay-summary.pdf"
    _write_pay_summary_pdf(
        pay_summary_path,
        report_date="06/30/26",
        salary=100_000.00,
        restricted_stock=1_000.00,
        federal_withholding=4_000.00,
        ca_withholding=1_000.00,
        medicare_withholding=1_464.50,
    )

    result = CliRunner().invoke(
        calc_taxes.app,
        [
            "vest",
            "--transactions-path",
            str(transactions_path),
            "--pay-summary-path",
            str(pay_summary_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Recommended IRS prepayment: $1,000.00" in result.output
    assert "Recommended CA FTB prepayment: $491.38" in result.output
    assert "standard deductions" in result.output.lower()
    assert "project" not in result.output.lower()
    assert "progressive tax mode" not in result.output.lower()


def test_vest_cli_total_income_overrides_both_income_tax_bases(
    tmp_path: Path,
) -> None:
    """Catch ignoring the override or applying it to one jurisdiction."""
    transactions_path = tmp_path / "transactions.csv"
    pd.DataFrame(
        [
            {
                "Date": "12/16/2026",
                "Action": "Lapse",
                "Symbol": "NVDA",
                "Quantity": 10.0,
                "FairMarketValuePrice": None,
                "SharesSoldWithheldForTaxes": None,
                "NetSharesDeposited": None,
                "Taxes": None,
            },
            {
                "Date": None,
                "Action": None,
                "Symbol": None,
                "Quantity": None,
                "FairMarketValuePrice": "$100.00",
                "SharesSoldWithheldForTaxes": 4.0,
                "NetSharesDeposited": 6.0,
                "Taxes": "$400.00",
            },
        ]
    ).to_csv(transactions_path, index=False)

    pay_summary_path = tmp_path / "pay-summary.pdf"
    _write_pay_summary_pdf(
        pay_summary_path,
        salary=100_000.00,
        restricted_stock=1_000.00,
        federal_withholding=4_000.00,
        ca_withholding=1_000.00,
        medicare_withholding=1_464.50,
    )

    result = CliRunner().invoke(
        calc_taxes.app,
        [
            "vest",
            "--transactions-path",
            str(transactions_path),
            "--pay-summary-path",
            str(pay_summary_path),
            "--total-income",
            "300000",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "User-provided total income: $300,000.00" in result.output
    assert (
        "Estimated taxable compensation before standard deduction: "
        "$300,000.00" in result.output
    )
    assert "Estimated federal income tax: $49,468.00" in result.output
    assert "Federal income tax withheld: $4,000.00" in result.output
    assert "Additional Medicare Tax: $0.00" in result.output
    assert (
        "Estimated California income before standard deduction: "
        "$300,000.00" in result.output
    )
    assert "Estimated California income tax: $19,715.96" in result.output
    assert "California income tax withheld: $1,000.00" in result.output
    assert "employer-only estimate" not in result.output


@pytest.mark.parametrize("total_income", ["-1", "nan", "inf"])
def test_vest_cli_rejects_invalid_total_income(
    tmp_path: Path,
    total_income: str,
) -> None:
    """Catch producing tax recommendations from an invalid income base."""
    transactions_path = tmp_path / "transactions.csv"
    pd.DataFrame(
        [
            {
                "Date": "12/16/2026",
                "Action": "Lapse",
                "Symbol": "NVDA",
                "Quantity": 10.0,
                "FairMarketValuePrice": None,
                "SharesSoldWithheldForTaxes": None,
                "NetSharesDeposited": None,
                "Taxes": None,
            },
            {
                "Date": None,
                "Action": None,
                "Symbol": None,
                "Quantity": None,
                "FairMarketValuePrice": "$100.00",
                "SharesSoldWithheldForTaxes": 4.0,
                "NetSharesDeposited": 6.0,
                "Taxes": "$400.00",
            },
        ]
    ).to_csv(transactions_path, index=False)
    pay_summary_path = tmp_path / "pay-summary.pdf"
    _write_pay_summary_pdf(
        pay_summary_path,
        restricted_stock=1_000.00,
    )

    result = CliRunner().invoke(
        calc_taxes.app,
        [
            "vest",
            "--transactions-path",
            str(transactions_path),
            "--pay-summary-path",
            str(pay_summary_path),
            "--total-income",
            total_income,
        ],
    )

    assert result.exit_code == 2
    assert "finite, nonnegative number" in result.output


def test_vest_cli_rejects_schwab_export_without_pay_summary_year(
    tmp_path: Path,
) -> None:
    """Catch silently using vest transactions from another tax year."""
    transactions_path = tmp_path / "transactions.csv"
    pd.DataFrame(
        [
            {
                "Date": "12/16/2025",
                "Action": "Lapse",
                "Symbol": "NVDA",
                "Quantity": 10.0,
                "FairMarketValuePrice": None,
                "SharesSoldWithheldForTaxes": None,
                "NetSharesDeposited": None,
                "Taxes": None,
            },
            {
                "Date": None,
                "Action": None,
                "Symbol": None,
                "Quantity": None,
                "FairMarketValuePrice": "$100.00",
                "SharesSoldWithheldForTaxes": 4.0,
                "NetSharesDeposited": 6.0,
                "Taxes": "$400.00",
            },
        ]
    ).to_csv(transactions_path, index=False)
    pay_summary_path = tmp_path / "pay-summary.pdf"
    _write_pay_summary_pdf(
        pay_summary_path,
        restricted_stock=1_000.00,
    )

    result = CliRunner().invoke(
        calc_taxes.app,
        [
            "vest",
            "--transactions-path",
            str(transactions_path),
            "--pay-summary-path",
            str(pay_summary_path),
        ],
    )

    assert result.exit_code == 2
    assert "no vest transactions for 2026" in result.output


def test_vest_cli_help_excludes_legacy_modes_and_paycheck() -> None:
    """Catch restoring manual tax modes or the unnecessary paycheck input."""
    result = CliRunner().invoke(calc_taxes.app, ["vest", "--help"])

    assert result.exit_code == 0
    assert "--transactions-path" in result.output
    assert "--pay-summary-path" in result.output
    assert "--total-income" in result.output
    for removed_option in [
        "--progressive-tax",
        "--other-income",
        "--withholding-rate",
        "--target-rate",
        "--ca-withholding-rate",
        "--ca-target-rate",
        "--paycheck-path",
    ]:
        assert removed_option not in result.output
