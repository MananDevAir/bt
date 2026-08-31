"""Backtest report — generates a readable summary from replay results.

Produces both console output and a markdown artifact.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from .replay import BacktestResult, BacktestSignal

__all__ = ["print_report", "save_report"]

IST = timezone(timedelta(hours=5, minutes=30))


def print_report(result: BacktestResult) -> None:
    """Print a formatted backtest report to console."""
    print(f"\n  {'=' * 70}")
    print(f"  BACKTEST REPORT — {result.symbol}")
    print(f"  {'=' * 70}")
    print(f"  Period:     {result.start_date} → {result.end_date}")
    print(f"  Step:       {result.step}")
    print(f"  Bars:       {result.bars_walked:,}")
    print(f"  Duration:   {result.duration_s:.1f}s")
    print()

    # Signal summary
    print(f"  SIGNALS: {result.total}")
    print(f"  {'-' * 50}")
    entered = len(result.entered)
    print(f"  Entry filled:  {entered}")
    print(f"  No entry:      {result.no_entry}")
    print(f"  Won:           {result.won}")
    print(f"  Lost:          {result.lost}")
    print(f"  Expired:       {result.expired}")
    print()

    # Win rate
    if result.won + result.lost > 0:
        wr = result.win_rate
        wr_icon = "🟢" if wr >= 55 else "🟡" if wr >= 45 else "🔴"
        print(f"  {wr_icon} WIN RATE: {wr:.1f}%  ({result.won}W / {result.lost}L)")
    else:
        print(f"  ⚪ No closed trades to compute win rate")
    print()

    # R metrics
    if entered > 0:
        print(f"  R METRICS:")
        print(f"  {'-' * 50}")
        print(f"  Avg MFE:   {result.avg_mfe_r:.2f}R")
        print(f"  Avg MAE:   {result.avg_mae_r:.2f}R")
        print(f"  Best:      {result.best_r:.2f}R")
        print(f"  Worst:    -{result.worst_r:.2f}R")
        edge = result.avg_mfe_r - result.avg_mae_r
        print(f"  Edge:      {edge:+.2f}R  {'✅ positive' if edge > 0 else '⚠️ negative'}")
        print()

    # By label
    by_label = result.by_label
    if by_label:
        print(f"  BY SIGNAL TYPE:")
        print(f"  {'-' * 50}")
        for label, data in sorted(by_label.items()):
            w, l, t = data["won"], data["lost"], data["total"]
            wr = w / (w + l) * 100 if (w + l) > 0 else 0
            wr_str = f"{wr:.0f}%" if (w + l) > 0 else "n/a"
            print(f"    {label:20s}  {t:>3d} signals  {w}W/{l}L  ({wr_str})")
        print()

    # By trade type
    by_type = result.by_trade_type
    if by_type:
        print(f"  BY TRADE TYPE:")
        print(f"  {'-' * 50}")
        for ttype, data in sorted(by_type.items()):
            w, l, t = data["won"], data["lost"], data["total"]
            wr = w / (w + l) * 100 if (w + l) > 0 else 0
            wr_str = f"{wr:.0f}%" if (w + l) > 0 else "n/a"
            print(f"    {ttype:20s}  {t:>3d} signals  {w}W/{l}L  ({wr_str})")
        print()

    # By symbol (if ALL)
    if result.symbol == "ALL":
        by_sym: dict[str, dict] = {}
        for s in result.signals:
            d = by_sym.setdefault(s.symbol, {"total": 0, "won": 0, "lost": 0})
            d["total"] += 1
            if s.outcome == "won":
                d["won"] += 1
            elif s.outcome == "lost":
                d["lost"] += 1

        if by_sym:
            print(f"  BY SYMBOL:")
            print(f"  {'-' * 50}")
            for sym, data in sorted(by_sym.items()):
                w, l, t = data["won"], data["lost"], data["total"]
                wr = w / (w + l) * 100 if (w + l) > 0 else 0
                wr_str = f"{wr:.0f}%" if (w + l) > 0 else "n/a"
                dot = "🟢" if wr >= 55 else "🔴" if (w + l) > 0 else "⚪"
                print(f"    {dot} {sym:10s}  {t:>3d} signals  {w}W/{l}L  ({wr_str})")
            print()

    # Signal details
    if result.signals:
        print(f"  SIGNAL LOG:")
        print(f"  {'-' * 70}")
        print(f"  {'Date':12s}  {'Symbol':8s}  {'Dir':5s}  {'Score':>6s}  "
              f"{'Type':12s}  {'RR':>4s}  {'Result':8s}  {'Hit':4s}  "
              f"{'MFE':>5s}  {'MAE':>5s}")
        print(f"  {'-' * 70}")
        for s in result.signals:
            dt = datetime.fromtimestamp(s.ts / 1000, tz=IST).strftime("%m-%d %H:%M")
            direction_str = "long" if s.direction > 0 else "short"
            icon = "✅" if s.outcome == "won" else "❌" if s.outcome == "lost" else "⏳"
            print(f"  {dt:12s}  {s.symbol:8s}  {direction_str:5s}  {s.score:+6.1f}  "
                  f"{s.trade_type:12s}  {s.rr:4.1f}  {icon} {s.outcome:6s}  "
                  f"{s.hit:4s}  {s.mfe_r:5.2f}  {s.mae_r:5.2f}")
        print()


def save_report(result: BacktestResult, path: Path) -> Path:
    """Save a markdown report to a file."""
    lines: list[str] = []

    ist_now = datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")
    lines.append(f"# Backtest Report — {result.symbol}")
    lines.append(f"")
    lines.append(f"> Generated: {ist_now}")
    lines.append(f"> Period: {result.start_date} → {result.end_date}")
    lines.append(f"> Step: {result.step}  |  Bars: {result.bars_walked:,}  |  Duration: {result.duration_s:.1f}s")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total signals | {result.total} |")
    lines.append(f"| Entry filled | {len(result.entered)} |")
    lines.append(f"| Won | {result.won} |")
    lines.append(f"| Lost | {result.lost} |")
    lines.append(f"| Expired | {result.expired} |")
    lines.append(f"| No entry | {result.no_entry} |")
    wr = result.win_rate
    wr_emoji = "🟢" if wr >= 55 else "🟡" if wr >= 45 else "🔴"
    lines.append(f"| **Win Rate** | **{wr_emoji} {wr:.1f}%** |")
    lines.append("")

    # R metrics
    if result.entered:
        lines.append("## Risk Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Avg MFE | {result.avg_mfe_r:.2f}R |")
        lines.append(f"| Avg MAE | {result.avg_mae_r:.2f}R |")
        lines.append(f"| Best | {result.best_r:.2f}R |")
        lines.append(f"| Worst | -{result.worst_r:.2f}R |")
        edge = result.avg_mfe_r - result.avg_mae_r
        lines.append(f"| **Edge** | **{edge:+.2f}R** |")
        lines.append("")

    # By symbol
    if result.symbol == "ALL":
        by_sym: dict[str, dict] = {}
        for s in result.signals:
            d = by_sym.setdefault(s.symbol, {"total": 0, "won": 0, "lost": 0})
            d["total"] += 1
            if s.outcome == "won":
                d["won"] += 1
            elif s.outcome == "lost":
                d["lost"] += 1

        if by_sym:
            lines.append("## By Symbol")
            lines.append("")
            lines.append("| Symbol | Signals | Won | Lost | Win Rate |")
            lines.append("|---|---|---|---|---|")
            for sym, data in sorted(by_sym.items()):
                w, l, t = data["won"], data["lost"], data["total"]
                wr = w / (w + l) * 100 if (w + l) > 0 else 0
                wr_str = f"{wr:.0f}%" if (w + l) > 0 else "—"
                lines.append(f"| {sym} | {t} | {w} | {l} | {wr_str} |")
            lines.append("")

    # By label
    by_label = result.by_label
    if by_label:
        lines.append("## By Signal Type")
        lines.append("")
        lines.append("| Label | Signals | Won | Lost | Win Rate |")
        lines.append("|---|---|---|---|---|")
        for label, data in sorted(by_label.items()):
            w, l, t = data["won"], data["lost"], data["total"]
            wr = w / (w + l) * 100 if (w + l) > 0 else 0
            wr_str = f"{wr:.0f}%" if (w + l) > 0 else "—"
            lines.append(f"| {label} | {t} | {w} | {l} | {wr_str} |")
        lines.append("")

    # Signal log
    if result.signals:
        lines.append("## Signal Log")
        lines.append("")
        lines.append("| Date | Symbol | Dir | Score | Type | RR | Result | Hit | MFE | MAE |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for s in result.signals:
            dt = datetime.fromtimestamp(s.ts / 1000, tz=IST).strftime("%m-%d %H:%M")
            direction_str = "long" if s.direction > 0 else "short"
            icon = "✅" if s.outcome == "won" else "❌" if s.outcome == "lost" else "⏳"
            lines.append(
                f"| {dt} | {s.symbol} | {direction_str} | {s.score:+.1f} | "
                f"{s.trade_type} | {s.rr:.1f} | {icon} {s.outcome} | "
                f"{s.hit} | {s.mfe_r:.2f} | {s.mae_r:.2f} |"
            )
        lines.append("")

    report_text = "\n".join(lines)
    path.write_text(report_text, encoding="utf-8")
    return path
