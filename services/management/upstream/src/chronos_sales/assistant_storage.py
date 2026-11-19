from __future__ import annotations

import csv
import io
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable


DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
MIN_HISTORY_DAYS = 28
MAX_CONTEXT_DAYS = 28


class RevenueDataError(ValueError):
    """Raised when revenue data cannot safely be used for forecasting."""


@dataclass(frozen=True)
class RevenueRecord:
    business_id: str
    date: date
    revenue: int
    source: str
    updated_at: str


@dataclass(frozen=True)
class ImportResult:
    inserted: int
    conflicts: tuple[date, ...]


def _parse_date(value: str) -> date:
    value = value.strip()
    if not DATE_PATTERN.fullmatch(value):
        raise RevenueDataError(f"날짜 '{value}'는 YYYY-MM-DD 형식이 아닙니다.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise RevenueDataError(f"존재하지 않는 날짜입니다: {value}") from exc


def _parse_revenue(value: str) -> int:
    value = value.strip()
    if not re.fullmatch(r"\d+", value):
        raise RevenueDataError("매출은 0 이상의 정수 KRW여야 합니다.")
    amount = int(value)
    if amount > 9_223_372_036_854_775_807:
        raise RevenueDataError("매출이 SQLite 정수 범위를 넘었습니다.")
    return amount


def parse_revenue_csv(raw: bytes | str) -> list[tuple[date, int]]:
    """Parse a strict date,revenue CSV without accepting duplicate dates."""
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise RevenueDataError("CSV는 UTF-8 인코딩이어야 합니다.") from exc
    else:
        text = raw.lstrip("\ufeff")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != ["date", "revenue"]:
        raise RevenueDataError("CSV 헤더는 date,revenue 두 열로 고정됩니다.")
    records: list[tuple[date, int]] = []
    seen: set[date] = set()
    for line_number, row in enumerate(reader, start=2):
        if None in row or row.get("date") is None or row.get("revenue") is None:
            raise RevenueDataError(f"CSV {line_number}행의 열 개수가 올바르지 않습니다.")
        try:
            parsed_date = _parse_date(row["date"])
            amount = _parse_revenue(row["revenue"])
        except RevenueDataError as exc:
            raise RevenueDataError(f"CSV {line_number}행: {exc}") from exc
        if parsed_date in seen:
            raise RevenueDataError(f"CSV에 중복 날짜가 있습니다: {parsed_date.isoformat()}")
        seen.add(parsed_date)
        records.append((parsed_date, amount))
    if not records:
        raise RevenueDataError("CSV에 등록할 매출이 없습니다.")
    return records


class RevenueStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_revenue (
                    business_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    revenue INTEGER NOT NULL CHECK (revenue >= 0),
                    source TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (business_id, date)
                )
                """
            )

    def import_csv(self, raw: bytes | str, business_id: str = "default") -> ImportResult:
        records = parse_revenue_csv(raw)
        existing_dates = {record.date for record in self.list_records(business_id)}
        conflicts = tuple(sorted(day for day, _ in records if day in existing_dates))
        new_records = [(day, amount) for day, amount in records if day not in existing_dates]
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO daily_revenue VALUES (?, ?, ?, ?, ?)",
                [
                    (business_id, day.isoformat(), amount, "csv", now)
                    for day, amount in new_records
                ],
            )
        return ImportResult(inserted=len(new_records), conflicts=conflicts)

    def upsert_manual(
        self, day: date, revenue: int, business_id: str = "default"
    ) -> None:
        if (
            isinstance(revenue, bool)
            or not isinstance(revenue, int)
            or revenue < 0
            or revenue > 9_223_372_036_854_775_807
        ):
            raise RevenueDataError("매출은 0 이상의 정수 KRW여야 합니다.")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO daily_revenue VALUES (?, ?, ?, 'manual', ?)
                ON CONFLICT(business_id, date) DO UPDATE SET
                    revenue = excluded.revenue,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (business_id, day.isoformat(), revenue, now),
            )

    def list_records(self, business_id: str = "default") -> list[RevenueRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT business_id, date, revenue, source, updated_at
                FROM daily_revenue WHERE business_id = ? ORDER BY date
                """,
                (business_id,),
            ).fetchall()
        return [
            RevenueRecord(
                business_id=row["business_id"],
                date=date.fromisoformat(row["date"]),
                revenue=int(row["revenue"]),
                source=row["source"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]


def validate_forecast_history(
    records: Iterable[RevenueRecord],
    min_days: int = MIN_HISTORY_DAYS,
    max_context: int = MAX_CONTEXT_DAYS,
) -> list[RevenueRecord]:
    ordered = sorted(records, key=lambda record: record.date)
    if len(ordered) < min_days:
        raise RevenueDataError(
            f"예측에는 연속 {min_days}일의 매출이 필요합니다. "
            f"현재 {len(ordered)}일입니다."
        )
    duplicate_dates = len({record.date for record in ordered}) != len(ordered)
    if duplicate_dates:
        raise RevenueDataError("중복 날짜가 있어 예측할 수 없습니다.")
    missing: list[date] = []
    for previous, current in zip(ordered, ordered[1:]):
        gap = (current.date - previous.date).days
        if gap != 1:
            missing.extend(
                previous.date.fromordinal(previous.date.toordinal() + offset)
                for offset in range(1, gap)
            )
    if missing:
        preview = ", ".join(day.isoformat() for day in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise RevenueDataError(
            f"누락일을 자동 0원 처리하지 않습니다. "
            f"매출 또는 휴무일 0원을 등록하세요: {preview}{suffix}"
        )
    return ordered[-max_context:]
