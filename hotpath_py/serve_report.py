"""Rich + plotext compact terminal renderer for hotpath serve-report.

Color philosophy (k9s / bottom / htop inspired):
  - Labels dim, values bright — eyes go straight to numbers
  - p50 dim gray, p90 normal, p99 bold + semantic color
  - Cyan for headers/identity, green=good, yellow=warn, red=bad
  - Thin dim rules separate sections without fighting content

Exit codes:  0 = success  1 = error  2 = deps missing (fall back to C++)
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

try:
    import plotext as plt
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


# ── palette ───────────────────────────────────────────────────────────────────
#  Inspired by k9s (cyan labels, semantic green/yellow/red) and bottom/htop
#  (bright values on dim labels, very subtle separators).

HI   = "bold cyan"          # identity / section labels
G    = "bright_green"       # good
Y    = "yellow"             # warn
R    = "bright_red"         # critical
M    = "magenta"            # prefill phase
C    = "cyan"               # GPU / info
DM   = "color(237)"         # barely-visible rule lines
BK   = "color(241)"         # secondary labels (dim)
P50  = "color(244)"         # p50 values — least prominent
P90  = "color(249)"         # p90 values — medium
TX   = "color(253)"         # primary text / p99 default
BR   = "bold white"         # highest emphasis
GD   = "bold bright_green"  # good + emphasis
YD   = "bold yellow"        # warn + emphasis
RD   = "bold bright_red"    # crit + emphasis


# ── DB ────────────────────────────────────────────────────────────────────────

def _kv(db: str) -> dict[str, str]:
    with sqlite3.connect(db) as con:
        if con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='serve_analysis'"
        ).fetchone() is None:
            return {}
        return {k: v for k, v in con.execute("SELECT key,value FROM serve_analysis").fetchall()}

def _ts(db: str, like: str) -> list[tuple[float, float]]:
    try:
        with sqlite3.connect(db) as con:
            return [(float(t), float(v)) for t, v in con.execute(
                "SELECT sample_time,value FROM vllm_metrics WHERE metric LIKE ? ORDER BY sample_time",
                (f"%{like}%",),
            ).fetchall()]
    except Exception:
        return []


# ── accessors ─────────────────────────────────────────────────────────────────

def _f(kv: dict, k: str, d: float = -1.0) -> float:
    try: return float(kv.get(k) or d)
    except: return d

def _i(kv: dict, k: str, d: int = 0) -> int:
    try: return int(kv.get(k) or d)
    except: return d

def _s(kv: dict, k: str, d: str = "") -> str:
    return kv.get(k, d)

def _b(kv: dict, k: str) -> bool:
    return kv.get(k) == "true"


# ── formatting ────────────────────────────────────────────────────────────────

def _ms(v: float) -> str:
    """Format a millisecond value with unit. Always shows unit."""
    if v < 0:   return "—"
    if v < 1:   return f"{v * 1000:.0f} µs"
    if v < 10:  return f"{v:.2f} ms"
    return f"{v:.1f} ms"

def _p99_style(v: float, warn: float, crit: float) -> str:
    """Bold + color for p99 — most important column."""
    if v < 0:        return BK
    if v >= crit:    return RD
    if v >= warn:    return YD
    return GD

BAR = "█"; EMPTY = "░"

def _bar(pct: float, w: int, color: str) -> Text:
    n = round(max(0.0, min(100.0, pct)) / 100 * w)
    t = Text()
    t.append(BAR * n, style=color)
    t.append(EMPTY * (w - n), style=DM)
    return t


# ── chart ─────────────────────────────────────────────────────────────────────

def _chart(times: list[float], vals: list[float], color: str,
           step: bool = False, w: int = 38, h: int = 8) -> str:
    if not times: return "(no data)"
    t0 = times[0]
    times = [t - t0 for t in times]
    plt.clf()
    plt.theme("dark")
    plt.plotsize(w, h)
    plt.canvas_color("default")
    plt.axes_color("default")
    plt.ticks_color("default")
    if step:
        st, sv = [], []
        for i, (t, v) in enumerate(zip(times, vals)):
            st.append(t); sv.append(v)
            if i < len(times) - 1:
                st.append(times[i + 1] - 0.001); sv.append(v)
        plt.plot(st, sv, color=color)
    else:
        plt.plot(times, vals, color=color)
    try: return plt.build()
    except: return "(chart error)"


# ── render ────────────────────────────────────────────────────────────────────

def render(db_path: str) -> int:
    if not DEPS_AVAILABLE:
        return 2

    if not Path(db_path).exists():
        print(f"error: file not found: {db_path}", file=sys.stderr)
        return 1

    kv = _kv(db_path)
    if not kv:
        print(f"error: no serve analysis in {db_path} — run serve-profile first",
              file=sys.stderr)
        return 1

    con = Console(highlight=False)

    # ── header ───────────────────────────────────────────────────────────────
    model  = _s(kv, "meta.model", "?")
    engine = _s(kv, "meta.engine", "?")
    gpu    = f"{_i(kv,'meta.gpu_count',1)}x {_s(kv,'meta.gpu_name','GPU')}"
    h = Text()
    h.append("hotpath", style="bold bright_green")
    h.append("  ")
    h.append(model,  style=BR)
    h.append("  ·  ", style=DM)
    h.append(engine, style=BK)
    h.append("  ·  ", style=DM)
    h.append(gpu,    style=C)
    con.print(h)

    # ── summary ───────────────────────────────────────────────────────────────
    req = _i(kv, "meta.total_requests")
    dur = _f(kv, "meta.duration_seconds")
    rps = _f(kv, "meta.throughput_rps")
    s = Text()
    s.append(f"{req:,}", style=BR);           s.append(" req  ", style=BK)
    s.append(f"{dur:.1f}", style=TX);          s.append("s  ",   style=BK)
    s.append(f"{rps:.1f}", style=GD);          s.append(" req/s", style=BK)
    if _b(kv, "latency.server_timing_available"): s.append("  srv-log",   style=DM)
    if _b(kv, "prefix.available"):               s.append("  pfx-cache", style=DM)
    con.print(s)
    con.rule(style=DM)

    # ── latency table ─────────────────────────────────────────────────────────
    # k9s-style: dim label column, p50 dimmer → p90 normal → p99 bold+colored
    lat = Table(show_header=True, box=None, padding=(0, 1), show_edge=False,
                header_style=BK)
    lat.add_column("",     no_wrap=True, min_width=22, style=BK)
    lat.add_column("p50",  justify="right", no_wrap=True, min_width=10)
    lat.add_column("p90",  justify="right", no_wrap=True, min_width=10)
    lat.add_column("p99 ▴", justify="right", no_wrap=True, min_width=11,
                   header_style="bold " + BK)  # ▴ flags it as the critical column

    def _row(label, p50, p90, p99, em=False, warn=200., crit=600.):
        if p50 < 0 and p99 < 0: return
        lbl = Text()
        if em:
            lbl.append("▸ ", style=G)
            lbl.append(label, style=BR)
        else:
            lbl.append(label, style=BK)
        lat.add_row(
            lbl,
            Text(_ms(p50), style=P50 if p50 >= 0 else DM),
            Text(_ms(p90), style=P90 if p90 >= 0 else DM),
            Text(_ms(p99), style=_p99_style(p99, warn, crit)),
        )

    if _b(kv, "latency.queue_available"):
        _row("queue wait",
             _f(kv, "latency.queue_p50"), _f(kv, "latency.queue_p90"),
             _f(kv, "latency.queue_p99"), warn=20, crit=50)
    if _b(kv, "latency.server_timing_available"):
        _row("prefill (server)",
             _f(kv, "latency.server_prefill_p50"), _f(kv, "latency.server_prefill_p90"),
             _f(kv, "latency.server_prefill_p99"), warn=50, crit=150)
        _row("decode (server)",
             _f(kv, "latency.server_decode_p50"), _f(kv, "latency.server_decode_p90"),
             _f(kv, "latency.server_decode_p99"), warn=300, crit=700)
        _row("decode / token",
             _f(kv, "latency.decode_per_token_p50"), _f(kv, "latency.decode_per_token_p90"),
             _f(kv, "latency.decode_per_token_p99"), warn=10, crit=20)
    else:
        _row("prefill",
             _f(kv, "latency.prefill_p50"), _f(kv, "latency.prefill_p90"),
             _f(kv, "latency.prefill_p99"), warn=50, crit=150)
        _row("decode",
             _f(kv, "latency.decode_total_p50"), _f(kv, "latency.decode_total_p90"),
             _f(kv, "latency.decode_total_p99"), warn=300, crit=700)
    _row("end-to-end",
         _f(kv, "latency.e2e_p50"), _f(kv, "latency.e2e_p90"),
         _f(kv, "latency.e2e_p99"), em=True, warn=400, crit=800)
    con.print(lat)

    # ── GPU phase ─────────────────────────────────────────────────────────────
    if _b(kv, "phase.available"):
        pre = _f(kv, "phase.prefill_pct", 0)
        dec = _f(kv, "phase.decode_pct",  0)
        oth = _f(kv, "phase.other_pct",   0)
        sch = max(0.0, 100 - pre - dec - oth)
        con.rule(style=DM)
        pt = Table.grid(padding=(0, 1))
        pt.add_column(min_width=10, style=BK)
        pt.add_column(min_width=5,  justify="right")
        pt.add_column()
        for name, pct, col in [("prefill", pre, M), ("decode", dec, G),
                                ("schedule", sch, Y), ("idle", oth, BK)]:
            pt.add_row(Text(name, style=BK),
                       Text(f"{pct:.0f}%", style=f"bold {col}"),
                       _bar(pct, 22, col))
        con.print(pt)

    # ── charts ────────────────────────────────────────────────────────────────
    tput_s  = _ts(db_path, "generation_throughput")
    batch_s = _ts(db_path, "num_requests_running")
    if tput_s or batch_s:
        con.rule(style=DM)
        CW = 38
        tc = _chart([r[0] for r in tput_s],  [r[1] for r in tput_s],
                    "green",  w=CW, h=8) if tput_s  else None
        bc = _chart([r[0] for r in batch_s], [r[1] for r in batch_s],
                    "cyan", step=True, w=CW, h=8) if batch_s else None
        panels = []
        if tc: panels.append(Panel(Text.from_ansi(tc.rstrip()),
                                   title="[color(241)]throughput tok/s[/]",
                                   border_style=DM, title_align="left", padding=(0, 0)))
        if bc: panels.append(Panel(Text.from_ansi(bc.rstrip()),
                                   title="[color(241)]batch size[/]",
                                   border_style=DM, title_align="left", padding=(0, 0)))
        if panels:
            con.print(Columns(panels, equal=True))

    # ── KV cache + prefix sharing ─────────────────────────────────────────────
    con.rule(style=DM)
    left  = Table.grid(padding=(0, 1))
    right = Table.grid(padding=(0, 1))
    for t in (left, right):
        t.add_column(style=BK, min_width=18)
        t.add_column(justify="right")

    if _b(kv, "cache.hit_rate_available"):
        hit = _f(kv, "cache.hit_rate", 0) * 100
        left.add_row("kv hit rate",
                     Text(f"{hit:.1f}%", style=GD if hit > 60 else (YD if hit > 30 else RD)))
    if _b(kv, "cache.usage_available"):
        avg_u  = _f(kv, "cache.avg_usage",  0) * 100
        peak_u = _f(kv, "cache.peak_usage", 0) * 100
        left.add_row("avg usage",   Text(f"{avg_u:.1f}%",  style=TX))
        left.add_row("peak usage",  Text(f"{peak_u:.1f}%", style=YD if peak_u > 90 else TX))
        left.add_row("evictions",   Text(str(_i(kv, "cache.evictions")), style=TX))

    if _b(kv, "prefix.available"):
        ctok = _f(kv, "prefix.cacheable_tokens_pct", 0)
        right.add_row("unique prefixes",  Text(str(_i(kv, "prefix.unique_prefixes")), style=TX))
        right.add_row("cacheable tokens", Text(f"{ctok:.1f}%", style=GD if ctok > 50 else YD))
        if _b(kv, "cache.hit_rate_available"):
            ahr = _f(kv, "cache.hit_rate", 0) * 100
            right.add_row("actual hit rate", Text(f"{ahr:.1f}%", style=GD if ahr > 60 else YD))

    if left.row_count or right.row_count:
        con.print(Columns([left, right], equal=True))

    # ── cache hit distribution ────────────────────────────────────────────────
    if _b(kv, "cache.histogram_available"):
        counts = [_i(kv, f"cache.histogram_{i}") for i in range(5)]
        total  = sum(counts) or 1
        mx     = max(counts) or 1
        ht = Table.grid(padding=(0, 1))
        ht.add_column(min_width=6,  justify="right", style=BK)
        ht.add_column(min_width=22)
        ht.add_column(min_width=12, justify="right", style=BK)
        for bucket, cnt, col in zip(["0%", "1-25%", "25-50%", "50-75%", "75%+"],
                                    counts, [BK, BK, C, G, GD]):
            ht.add_row(bucket, _bar(cnt / mx * 100, 20, col),
                       f"{cnt:,}  ({cnt / total * 100:.0f}%)")
        con.rule(style=DM)
        con.print(ht)

    # ── disaggregation advisor ────────────────────────────────────────────────
    con.rule(style=DM)
    should = _b(kv, "disagg.should")
    wc     = _s(kv, "disagg.workload_class")
    mono   = _f(kv, "disagg.mono_p99_ttft",        -1)
    disagg = _f(kv, "disagg.disagg_p99_ttft",      -1)
    proj   = _f(kv, "disagg.disagg_throughput",    -1)
    impr   = _f(kv, "disagg.throughput_improvement", -1)
    opt_p  = _i(kv, "disagg.optimal_p")
    opt_d  = _i(kv, "disagg.optimal_d")
    bw     = _f(kv, "disagg.min_bandwidth", -1)
    caveat = _s(kv, "disagg.caveat")

    da = Table.grid(padding=(0, 1))
    da.add_column(style=BK, min_width=20)
    da.add_column()

    rec = Text()
    if should:
        rec.append(" ✦ DISAGGREGATE ", style=f"bold reverse {G}")
    else:
        rec.append(" ✓ MONOLITHIC ",   style=f"bold reverse {Y}")
    da.add_row("advisor", rec)

    if wc:
        da.add_row("workload", Text(wc, style=C))
    if mono > 0 and disagg > 0:
        pct = (mono - disagg) / mono * 100
        da.add_row("prefill contention",
                   Text(f"+{pct:.0f}% on p99 latency", style=Y))
    if rps > 0 and proj > 0 and impr > 0:
        tpct = (impr - 1) * 100
        da.add_row("throughput",
                   Text(f"{rps:.1f}  →  {proj:.1f} req/s  (+{tpct:.0f}%)", style=GD))
    if mono > 0 and disagg > 0:
        d2 = (mono - disagg) / mono * 100
        da.add_row("p99 TTFT",
                   Text(f"{mono:.1f}  →  {disagg:.1f} ms  (-{d2:.0f}%)", style=GD))
    if opt_p > 0 and opt_d > 0:
        da.add_row("P/D ratio",
                   Text(f"1:{opt_d // opt_p}  ({opt_p}P + {opt_d}D)", style=TX))
    if bw > 0:
        da.add_row("min bandwidth",
                   Text(f"{bw:.0f} Gbps" + ("  ⚠ below this disagg hurts" if bw < 50 else ""),
                        style=YD if bw < 50 else TX))
    if caveat:
        da.add_row("note", Text(caveat.split(".")[0], style=DM))

    con.print(da)
    con.rule(style=DM)

    # ── footer ────────────────────────────────────────────────────────────────
    f = Text()
    f.append("hotpath disagg-config ", style=BK)
    f.append(Path(db_path).name, style=TX)
    f.append("  →  vLLM / llm-d / Dynamo", style=DM)
    con.print(f)

    return 0


if __name__ == "__main__":
    if not DEPS_AVAILABLE:
        sys.exit(2)
    if len(sys.argv) < 2:
        print("usage: python3 -m hotpath_py.serve_report <serve_profile.db>",
              file=sys.stderr)
        sys.exit(1)
    sys.exit(render(sys.argv[1]))
