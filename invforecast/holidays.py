"""Bangladesh holiday calendar loaded from YAML.

Fixed holidays recur yearly; moveable ones (Eid, Puja, ...) are listed per year as
configurable approximations. Eid entries carry a ``span`` of several days and a
run-up window during which retail demand climbs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "bd_holidays.yaml"


@dataclass(frozen=True)
class HolidayCalendar:
    holidays: dict[date, str] = field(default_factory=dict)
    eid_days: frozenset[date] = frozenset()
    eid_runup_days: int = 10

    def is_holiday(self, day: date) -> bool:
        return day in self.holidays

    def name(self, day: date) -> str | None:
        return self.holidays.get(day)

    def is_eid(self, day: date) -> bool:
        return day in self.eid_days

    def days_to_next_eid(self, day: date) -> int | None:
        future = sorted(d for d in self.eid_days if d >= day)
        return (future[0] - day).days if future else None

    def is_pre_eid(self, day: date) -> bool:
        delta = self.days_to_next_eid(day)
        return delta is not None and 0 < delta <= self.eid_runup_days


def _as_date(value) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def load_calendar(path: Path | str | None = None, years: range | None = None) -> HolidayCalendar:
    """Load the calendar; fixed holidays are expanded over ``years`` (default: listed years +-1)."""
    raw = yaml.safe_load(Path(path or DEFAULT_PATH).read_text(encoding="utf-8")) or {}
    holidays: dict[date, str] = {}
    eid_days: set[date] = set()

    moveable = raw.get("moveable") or {}
    listed_years = sorted(int(y) for y in moveable)
    this_year = date.today().year
    year_range = years or range(
        (listed_years[0] if listed_years else this_year) - 1,
        (listed_years[-1] if listed_years else this_year) + 2,
    )
    for year in year_range:
        for item in raw.get("fixed") or []:
            try:
                holidays[date(year, int(item["month"]), int(item["day"]))] = str(item["name"])
            except ValueError:
                continue

    for _year, items in moveable.items():
        for item in items or []:
            start = _as_date(item["date"])
            span = int(item.get("span", 1))
            for offset in range(span):
                day = start + timedelta(days=offset)
                holidays[day] = str(item["name"])
                if "eid" in str(item["name"]).lower():
                    eid_days.add(day)

    for item in raw.get("custom") or []:
        holidays[_as_date(item["date"])] = str(item.get("name", "Custom closure"))

    return HolidayCalendar(
        holidays=holidays,
        eid_days=frozenset(eid_days),
        eid_runup_days=int(raw.get("eid_runup_days", 10)),
    )
