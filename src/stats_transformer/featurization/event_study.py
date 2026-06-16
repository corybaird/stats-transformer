"""Event-study window builder for panel time-series data.

Generalizes the pattern in mpsd-rates-research's event_market_models.py:
given a value panel (entity, date, value) and an events frame (entity,
event_date), locate each event in the series and compute horizon changes
(log-pct or raw diff) from the pre-event observation.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EventStudyBuilder:
    """Compute h-horizon changes around event dates for a panel value series.

    Parameters
    ----------
    entity_col : str
        Column name for the entity identifier (e.g. 'country').
    date_col : str
        Column name for the date column in both the values and events frames.

    Examples
    --------
    >>> builder = EventStudyBuilder()
    >>> windows = builder.build(
    ...     values=fx_panel,
    ...     events=meetings,
    ...     value_col="usd_exchange_rate",
    ...     horizons=[1, 5, 22],
    ...     change="logpct",
    ...     scale=100.0,
    ... )
    """

    def __init__(
        self,
        entity_col: str = "country",
        date_col: str = "date",
    ) -> None:
        self.entity_col = entity_col
        self.date_col = date_col

    def build(
        self,
        values: pd.DataFrame,
        events: pd.DataFrame,
        *,
        value_col: str,
        horizons: list[int],
        change: Literal["logpct", "raw"] = "logpct",
        scale: float = 1.0,
        event_date_col: str = "event_date",
        col_prefix: str | None = None,
    ) -> pd.DataFrame:
        """Return one row per (entity, event) with h-horizon change columns.

        Parameters
        ----------
        values : pd.DataFrame
            Panel of (entity_col, date_col, value_col). Must be sortable by
            entity and date; duplicates on (entity, date) are deduplicated by
            keeping the last observation.
        events : pd.DataFrame
            Frame of events with at least (entity_col, event_date_col).
            All other columns are preserved in the output.
        value_col : str
            Column in ``values`` holding the numeric series.
        horizons : list[int]
            Business-day horizons h to compute: change from day-before-event
            to day+h (using array indices, not calendar days).
        change : "logpct" or "raw"
            "logpct" → 100 * log(v[idx+h] / v[idx-1]).
            "raw"    → scale * (v[idx+h] - v[idx-1]).
        scale : float
            Multiplier applied to "raw" changes (e.g. 100 for bps from pct).
        event_date_col : str
            Column in ``events`` holding the event date (default "event_date").
        col_prefix : str or None
            Prefix for horizon columns. Defaults to ``value_col``.

        Returns
        -------
        pd.DataFrame
            All columns from ``events``, plus
            ``{prefix}_event_date`` and ``{prefix}_d0_d{h}_change[_bps]``
            for each horizon h.  Rows without a matching pre-event observation
            in the values series are silently dropped.
        """
        prefix = col_prefix if col_prefix is not None else value_col
        suffix = "_bps" if change == "raw" and scale != 1.0 else ""

        values = values.copy()
        values[self.date_col] = pd.to_datetime(values[self.date_col], errors="coerce")
        values[value_col] = pd.to_numeric(values[value_col], errors="coerce")
        values = values.dropna(subset=[self.entity_col, self.date_col, value_col])

        events = events.copy()
        events[event_date_col] = pd.to_datetime(events[event_date_col], errors="coerce")
        events = events.dropna(subset=[self.entity_col, event_date_col])

        rows: list[dict] = []
        for entity, entity_events in events.groupby(self.entity_col):
            series = (
                values[values[self.entity_col] == entity]
                .drop_duplicates(self.date_col, keep="last")
                .sort_values(self.date_col)
            )
            if series.empty:
                continue

            dates = series[self.date_col].to_numpy()
            vals = series[value_col].to_numpy(dtype=float)

            for event in entity_events.itertuples(index=False):
                event_date = getattr(event, event_date_col)
                idx = int(np.searchsorted(dates, np.datetime64(event_date), side="left"))
                if idx <= 0 or idx >= len(vals):
                    continue

                row = event._asdict()
                row[f"{prefix}_event_date"] = pd.Timestamp(dates[idx])

                for h in horizons:
                    end_idx = min(idx + h, len(vals) - 1)
                    if change == "logpct":
                        delta = 100.0 * (np.log(vals[end_idx]) - np.log(vals[idx - 1]))
                    else:
                        delta = scale * (vals[end_idx] - vals[idx - 1])
                    row[f"{prefix}_d0_d{h}_change{suffix}"] = delta

                rows.append(row)

        out = pd.DataFrame(rows)
        logger.info(
            "EventStudyBuilder: entity=%s, value=%s, events=%d, matched=%d",
            self.entity_col, value_col, len(events), len(out),
        )
        return out
