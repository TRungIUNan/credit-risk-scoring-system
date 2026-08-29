"""Self-contained validation pipeline for the credit risk raw dataset.

Run from project root:
    python -m src.data.validation

Optional:
    python -m src.data.validation --data "data/raw/Credit Risk Dataset.xlsx"
    python -m src.data.validation --json-output reports/validation_results.json

The module intentionally uses only the Python standard library so the project
can satisfy the Data Audit completion check even before the modeling
environment is fully installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


TARGET_COL = "loan_status"
ID_COL = "client_ID"
DATA_SHEET_NAME = "Credit Risk Data"

EXPECTED_COLUMNS = [
    "client_ID",
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_default_on_file",
    "open_accounts",
    "gender",
    "marital_status",
    "education_level",
    "country",
    "state",
    "city",
    "city_latitude",
    "city_longitude",
    "employment_type",
    "loan_term_months",
    "credit_utilization_ratio",
    "other_debt",
    "cb_person_cred_hist_length",
    "loan_to_income_ratio",
    "debt_to_income_ratio",
    "past_delinquencies",
]

TEXT_MISSING_TOKENS = {
    "",
    "-",
    "?",
    "missing",
    "n/a",
    "na",
    "nan",
    "none",
    "null",
    "unknown",
}

EXPECTED_MISSING = {
    "person_emp_length": {"expected_rate": 0.0275, "tolerance": 0.0050},
    "loan_int_rate": {"expected_rate": 0.0956, "tolerance": 0.0050},
}

POTENTIAL_LEAKAGE_COLUMNS = {
    "loan_grade": "Potential post-underwriting risk grade.",
    "loan_int_rate": "Potential post-underwriting pricing or risk signal.",
}


@dataclass
class ValidationResult:
    check_name: str
    category: str
    severity: str
    status: str
    affected_columns: str
    affected_rows: int | None
    affected_rate: float | None
    message: str


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if all((candidate / folder).exists() for folder in ("data", "notebooks", "src")):
            return candidate
    raise FileNotFoundError("Cannot find CREDIT_RISK_PROJECT root.")


def resolve_data_path(path_arg: str | None) -> Path:
    project_root = find_project_root(Path.cwd().resolve())
    if path_arg:
        data_path = Path(path_arg)
        if not data_path.is_absolute():
            data_path = project_root / data_path
        return data_path.resolve()
    return (project_root / "data" / "raw" / "Credit Risk Dataset.xlsx").resolve()


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def cell_ref_to_col_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref)
    if not letters:
        raise ValueError(f"Invalid Excel cell reference: {cell_ref}")
    index = 0
    for char in letters.group(0):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def get_child_text(element: ET.Element, child_name: str) -> str | None:
    for child in element:
        if strip_namespace(child.tag) == child_name:
            return child.text
    return None


def read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for si in root:
        if strip_namespace(si.tag) != "si":
            continue
        parts: list[str] = []
        for node in si.iter():
            if strip_namespace(node.tag) == "t" and node.text is not None:
                parts.append(node.text)
        strings.append("".join(parts))
    return strings


def read_workbook_relationships(archive: zipfile.ZipFile) -> dict[str, str]:
    root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relationships: dict[str, str] = {}
    for rel in root:
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rel_id and target:
            relationships[rel_id] = "xl/" + target.lstrip("/")
    return relationships


def find_sheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = read_workbook_relationships(archive)

    for element in workbook_root.iter():
        if strip_namespace(element.tag) != "sheet":
            continue
        if element.attrib.get("name") != sheet_name:
            continue
        rel_id = (
            element.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            or element.attrib.get("r:id")
        )
        if rel_id not in rels:
            raise ValueError(f"Cannot resolve sheet relationship for {sheet_name!r}.")
        return rels[rel_id].replace("\\", "/")

    available = [
        element.attrib.get("name")
        for element in workbook_root.iter()
        if strip_namespace(element.tag) == "sheet"
    ]
    raise ValueError(f"Sheet {sheet_name!r} not found. Available sheets: {available}")


def parse_excel_cell(cell: ET.Element, shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t")
    value_text = get_child_text(cell, "v")

    if cell_type == "inlineStr":
        parts = [
            node.text
            for node in cell.iter()
            if strip_namespace(node.tag) == "t" and node.text is not None
        ]
        return "".join(parts)

    if value_text is None:
        return None

    if cell_type == "s":
        return shared_strings[int(value_text)]
    if cell_type == "b":
        return int(value_text)
    if cell_type == "str":
        return value_text

    try:
        number = float(value_text)
    except ValueError:
        return value_text
    if number.is_integer():
        return int(number)
    return number


def load_xlsx_rows(data_path: Path, sheet_name: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(data_path) as archive:
        shared_strings = read_shared_strings(archive)
        sheet_path = find_sheet_path(archive, sheet_name)
        sheet_root = ET.fromstring(archive.read(sheet_path))

    raw_rows: list[list[Any]] = []
    for row in sheet_root.iter():
        if strip_namespace(row.tag) != "row":
            continue
        values: list[Any] = []
        for cell in row:
            if strip_namespace(cell.tag) != "c":
                continue
            cell_ref = cell.attrib.get("r", "")
            col_index = cell_ref_to_col_index(cell_ref)
            while len(values) <= col_index:
                values.append(None)
            values[col_index] = parse_excel_cell(cell, shared_strings)
        raw_rows.append(values)

    if not raw_rows:
        return []

    headers = [str(value).strip() if value is not None else "" for value in raw_rows[0]]
    records: list[dict[str, Any]] = []
    for values in raw_rows[1:]:
        row = {
            header: values[index] if index < len(values) else None
            for index, header in enumerate(headers)
        }
        if any(value is not None and value != "" for value in row.values()):
            records.append(row)
    return records


def load_csv_rows(data_path: Path) -> list[dict[str, Any]]:
    with data_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_dataset(data_path: Path, sheet_name: str) -> list[dict[str, Any]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return load_xlsx_rows(data_path, sheet_name)
    if suffix == ".csv":
        return load_csv_rows(data_path)
    raise ValueError(f"Unsupported dataset format: {suffix}")


def columns_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    return list(rows[0].keys()) if rows else []


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value == "":
        return True
    return False


def to_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def make_result(
    check_name: str,
    category: str,
    severity: str,
    status: str,
    affected_columns: str,
    message: str,
    affected_rows: int | None = None,
    total_rows: int | None = None,
) -> ValidationResult:
    affected_rate = None
    if affected_rows is not None and total_rows:
        affected_rate = affected_rows / total_rows
    return ValidationResult(
        check_name=check_name,
        category=category,
        severity=severity,
        status=status,
        affected_columns=affected_columns,
        affected_rows=affected_rows,
        affected_rate=affected_rate,
        message=message,
    )


def check_schema(rows: list[dict[str, Any]]) -> list[ValidationResult]:
    columns = columns_from_rows(rows)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(columns))
    extra_columns = sorted(set(columns) - set(EXPECTED_COLUMNS))
    duplicate_column_count = len(columns) - len(set(columns))

    schema_ok = not missing_columns and not extra_columns and duplicate_column_count == 0
    result = make_result(
        check_name="schema_expected_columns",
        category="Schema",
        severity="low" if schema_ok else "high",
        status="passed" if schema_ok else "failed",
        affected_columns=", ".join(missing_columns + extra_columns) or "-",
        affected_rows=None,
        total_rows=len(rows),
        message=(
            "Schema matches expected Data Audit columns."
            if schema_ok
            else (
                f"Schema mismatch. Missing={missing_columns}; "
                f"extra={extra_columns}; duplicate_columns={duplicate_column_count}."
            )
        ),
    )

    required_results = []
    for column in (TARGET_COL, ID_COL):
        exists = column in columns
        required_results.append(
            make_result(
                check_name=f"required_column_{column}",
                category="Schema",
                severity="high",
                status="passed" if exists else "failed",
                affected_columns=column,
                affected_rows=None,
                total_rows=len(rows),
                message=(
                    f"Required column {column} exists."
                    if exists
                    else f"Required column {column} is missing."
                ),
            )
        )
    return [result, *required_results]


def check_duplicates(rows: list[dict[str, Any]]) -> list[ValidationResult]:
    columns = columns_from_rows(rows)
    row_tuples = [tuple(row.get(column) for column in columns) for row in rows]
    duplicate_row_count = len(row_tuples) - len(set(row_tuples))

    results = [
        make_result(
            check_name="duplicate_rows",
            category="Duplicate",
            severity="low" if duplicate_row_count == 0 else "medium",
            status="passed" if duplicate_row_count == 0 else "warning",
            affected_columns="All columns",
            affected_rows=duplicate_row_count,
            total_rows=len(rows),
            message=(
                "No duplicate rows found."
                if duplicate_row_count == 0
                else f"Found {duplicate_row_count:,} duplicate rows."
            ),
        )
    ]

    if ID_COL not in columns:
        results.append(
            make_result(
                check_name="duplicate_client_id",
                category="Duplicate",
                severity="high",
                status="failed",
                affected_columns=ID_COL,
                affected_rows=None,
                total_rows=len(rows),
                message=f"Cannot check duplicate {ID_COL}; column is missing.",
            )
        )
        return results

    seen: set[Any] = set()
    duplicate_ids: set[Any] = set()
    for row in rows:
        value = row.get(ID_COL)
        if value in seen:
            duplicate_ids.add(value)
        seen.add(value)
    duplicate_id_rows = sum(1 for row in rows if row.get(ID_COL) in duplicate_ids)

    results.append(
        make_result(
            check_name="duplicate_client_id",
            category="Duplicate",
            severity="low" if duplicate_id_rows == 0 else "medium",
            status="passed" if duplicate_id_rows == 0 else "warning",
            affected_columns=ID_COL,
            affected_rows=duplicate_id_rows,
            total_rows=len(rows),
            message=(
                f"{ID_COL} is unique."
                if duplicate_id_rows == 0
                else f"Found {duplicate_id_rows:,} rows with duplicated {ID_COL}."
            ),
        )
    )
    return results


def missing_count(rows: list[dict[str, Any]], column: str) -> int:
    return sum(1 for row in rows if is_missing(row.get(column)))


def check_missing_values(rows: list[dict[str, Any]]) -> list[ValidationResult]:
    columns = columns_from_rows(rows)
    missing_counts = {column: missing_count(rows, column) for column in columns}
    missing_columns = {column: count for column, count in missing_counts.items() if count > 0}
    unexpected_missing = [
        column for column in missing_columns if column not in EXPECTED_MISSING
    ]
    unexpected_count = sum(missing_columns[column] for column in unexpected_missing)

    results = [
        make_result(
            check_name="unexpected_missing_columns",
            category="Missing",
            severity="low" if not unexpected_missing else "medium",
            status="passed" if not unexpected_missing else "warning",
            affected_columns=", ".join(unexpected_missing) or "-",
            affected_rows=unexpected_count,
            total_rows=len(rows),
            message=(
                "No unexpected missing columns beyond Data Audit watchlist."
                if not unexpected_missing
                else f"Unexpected missing columns found: {unexpected_missing}."
            ),
        )
    ]

    for column, expectation in EXPECTED_MISSING.items():
        if column not in columns:
            results.append(
                make_result(
                    check_name=f"expected_missing_{column}",
                    category="Missing",
                    severity="high",
                    status="failed",
                    affected_columns=column,
                    affected_rows=None,
                    total_rows=len(rows),
                    message=f"Expected missing check cannot run; {column} is missing.",
                )
            )
            continue

        count = missing_counts[column]
        rate = count / len(rows) if rows else 0
        expected_rate = expectation["expected_rate"]
        tolerance = expectation["tolerance"]
        within_expected_range = abs(rate - expected_rate) <= tolerance
        results.append(
            make_result(
                check_name=f"expected_missing_{column}",
                category="Missing",
                severity="medium",
                status="passed" if within_expected_range else "warning",
                affected_columns=column,
                affected_rows=count,
                total_rows=len(rows),
                message=(
                    f"{column} missing rate is {rate:.2%}; "
                    f"expected about {expected_rate:.2%}."
                ),
            )
        )
    return results


def check_encoded_missing_tokens(rows: list[dict[str, Any]]) -> list[ValidationResult]:
    columns = columns_from_rows(rows)
    numeric_like_columns = {
        column
        for column in columns
        if all(is_missing(row.get(column)) or to_float(row.get(column)) is not None for row in rows)
    }
    text_columns = [column for column in columns if column not in numeric_like_columns]

    token_records: list[dict[str, Any]] = []
    for column in text_columns:
        counts: dict[str, int] = {}
        for row in rows:
            value = row.get(column)
            if not isinstance(value, str):
                continue
            normalized = value.strip().lower()
            if normalized in TEXT_MISSING_TOKENS:
                counts[normalized] = counts.get(normalized, 0) + 1
        if counts:
            token_records.append(
                {
                    "column": column,
                    "token_count": sum(counts.values()),
                    "tokens_found": sorted(counts),
                }
            )

    affected_rows = sum(record["token_count"] for record in token_records)
    return [
        make_result(
            check_name="encoded_missing_tokens",
            category="Invalid",
            severity="low" if not token_records else "medium",
            status="passed" if not token_records else "warning",
            affected_columns=", ".join(record["column"] for record in token_records) or "-",
            affected_rows=affected_rows,
            total_rows=len(rows),
            message=(
                "No encoded string missing tokens found."
                if not token_records
                else f"Encoded missing tokens found: {token_records}."
            ),
        )
    ]


def check_business_rules(rows: list[dict[str, Any]]) -> list[ValidationResult]:
    columns = columns_from_rows(rows)
    results: list[ValidationResult] = []

    if "person_age" in columns:
        age_count = 0
        for row in rows:
            age = to_float(row.get("person_age"))
            if age is not None and not (20 <= age <= 100):
                age_count += 1
        results.append(
            make_result(
                check_name="person_age_range",
                category="Range violation",
                severity="low" if age_count == 0 else "medium",
                status="passed" if age_count == 0 else "warning",
                affected_columns="person_age",
                affected_rows=age_count,
                total_rows=len(rows),
                message=(
                    "person_age is within [20, 100]."
                    if age_count == 0
                    else f"Found {age_count:,} rows where person_age is outside [20, 100]."
                ),
            )
        )
    else:
        results.append(
            make_result(
                check_name="person_age_range",
                category="Range violation",
                severity="high",
                status="failed",
                affected_columns="person_age",
                affected_rows=None,
                total_rows=len(rows),
                message="Cannot check person_age range; column is missing.",
            )
        )

    required_emp_cols = {"person_age", "person_emp_length"}
    if required_emp_cols.issubset(columns):
        min_working_age = 14
        emp_count = 0
        for row in rows:
            age = to_float(row.get("person_age"))
            emp_length = to_float(row.get("person_emp_length"))
            if emp_length is None:
                continue
            if emp_length < 0 or (age is not None and emp_length > age - min_working_age):
                emp_count += 1
        results.append(
            make_result(
                check_name="person_emp_length_vs_age",
                category="Range violation",
                severity="low" if emp_count == 0 else "medium",
                status="passed" if emp_count == 0 else "warning",
                affected_columns="person_emp_length, person_age",
                affected_rows=emp_count,
                total_rows=len(rows),
                message=(
                    f"person_emp_length is within [0, age - {min_working_age}]."
                    if emp_count == 0
                    else (
                        f"Found {emp_count:,} rows where person_emp_length is negative "
                        f"or greater than person_age - {min_working_age}."
                    )
                ),
            )
        )
    else:
        missing = sorted(required_emp_cols - set(columns))
        results.append(
            make_result(
                check_name="person_emp_length_vs_age",
                category="Range violation",
                severity="high",
                status="failed",
                affected_columns=", ".join(missing),
                affected_rows=None,
                total_rows=len(rows),
                message=f"Cannot check employment length rule; missing columns: {missing}.",
            )
        )

    if TARGET_COL in columns:
        invalid_target_count = 0
        for row in rows:
            value = row.get(TARGET_COL)
            if value not in (0, 1, "0", "1"):
                invalid_target_count += 1
        results.append(
            make_result(
                check_name="target_domain",
                category="Invalid",
                severity="high" if invalid_target_count else "low",
                status="failed" if invalid_target_count else "passed",
                affected_columns=TARGET_COL,
                affected_rows=invalid_target_count,
                total_rows=len(rows),
                message=(
                    f"{TARGET_COL} only contains 0/1."
                    if invalid_target_count == 0
                    else f"Found {invalid_target_count:,} rows where {TARGET_COL} is not 0/1."
                ),
            )
        )

    if "credit_utilization_ratio" in columns:
        util_count = 0
        for row in rows:
            value = to_float(row.get("credit_utilization_ratio"))
            if value is not None and not (0 <= value <= 1):
                util_count += 1
        results.append(
            make_result(
                check_name="credit_utilization_ratio_range",
                category="Range violation",
                severity="low" if util_count == 0 else "medium",
                status="passed" if util_count == 0 else "warning",
                affected_columns="credit_utilization_ratio",
                affected_rows=util_count,
                total_rows=len(rows),
                message=(
                    "credit_utilization_ratio is within [0, 1]."
                    if util_count == 0
                    else f"Found {util_count:,} rows outside [0, 1]."
                ),
            )
        )

    return results


def check_potential_leakage(rows: list[dict[str, Any]]) -> list[ValidationResult]:
    columns = columns_from_rows(rows)
    results: list[ValidationResult] = []
    for column, reason in POTENTIAL_LEAKAGE_COLUMNS.items():
        exists = column in columns
        results.append(
            make_result(
                check_name=f"potential_leakage_{column}",
                category="Potential leakage",
                severity="high" if exists else "low",
                status="watchlist" if exists else "passed",
                affected_columns=column,
                affected_rows=None,
                total_rows=len(rows),
                message=(
                    f"{column} exists and should be excluded from Primary Model until "
                    f"application-time availability is confirmed. Reason: {reason}"
                    if exists
                    else f"{column} is not present."
                ),
            )
        )
    return results


def run_validation(rows: list[dict[str, Any]]) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    results.extend(check_schema(rows))
    results.extend(check_duplicates(rows))
    results.extend(check_missing_values(rows))
    results.extend(check_encoded_missing_tokens(rows))
    results.extend(check_business_rules(rows))
    results.extend(check_potential_leakage(rows))
    return results


def format_rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def format_rows(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"


def print_results(results: list[ValidationResult]) -> None:
    headers = [
        "status",
        "severity",
        "category",
        "check_name",
        "affected_columns",
        "affected_rows",
        "affected_rate",
        "message",
    ]
    records = []
    for result in results:
        row = asdict(result)
        row["affected_rows"] = format_rows(result.affected_rows)
        row["affected_rate"] = format_rate(result.affected_rate)
        records.append(row)

    widths = {
        header: max(len(header), *(len(str(record[header])) for record in records))
        for header in headers
    }
    line = " | ".join(header.ljust(widths[header]) for header in headers)
    print(line)
    print("-+-".join("-" * widths[header] for header in headers))
    for record in records:
        print(" | ".join(str(record[header]).ljust(widths[header]) for header in headers))

    print()
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    print("Status counts:")
    for status, count in sorted(status_counts.items()):
        print(f"- {status}: {count}")


def write_json_report(results: list[ValidationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the credit risk raw dataset.")
    parser.add_argument(
        "--data",
        default=None,
        help="Path to dataset. Defaults to data/raw/Credit Risk Dataset.xlsx.",
    )
    parser.add_argument(
        "--sheet",
        default=DATA_SHEET_NAME,
        help=f"Excel sheet name. Defaults to {DATA_SHEET_NAME!r}.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path for JSON validation results.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return non-zero exit code when warnings/watchlist items are present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = resolve_data_path(args.data)
    rows = load_dataset(data_path, args.sheet)
    results = run_validation(rows)

    print(f"Dataset: {data_path}")
    print(f"Rows   : {len(rows):,}")
    print(f"Columns: {len(columns_from_rows(rows)):,}")
    print()
    print_results(results)

    if args.json_output:
        output_path = Path(args.json_output)
        if not output_path.is_absolute():
            output_path = find_project_root(Path.cwd().resolve()) / output_path
        write_json_report(results, output_path.resolve())
        print(f"JSON validation report saved to: {output_path.resolve()}")

    failed_count = sum(1 for result in results if result.status == "failed")
    warning_count = sum(1 for result in results if result.status in {"warning", "watchlist"})
    if failed_count:
        return 1
    if args.fail_on_warning and warning_count:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
