"""Rich + plotext terminal renderer for hotpath serve-report.

Can be invoked directly by the C++ binary:
    python3 -m hotpath_py.serve_report <db_path>

Exit codes:
    0  — rendered successfully
    1  — error (bad db, missing data)
    2  — rich/plotext not available; caller should fall back to C++ renderer
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any

try:
    import plotext as plt
    import rich
    from rich.columns import Columns
    from rich.console import Console
    from rich.padding import Padding
    from rich.style import Style
    from rich.table import Table
    from rich.text import Text

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


# ── colour palette ───────────────────────────────────────────────────────────
C_GREEN  = "bright_green"
C_AMBER  = "yellow"
C_RED    = "bright_red"
C_PURPLE = "magenta"
C_CYAN   = "bright_cyan"
C_MUTED  = "bright_black"
C_DIM    = "color(237)"
C_BODY   = "color(245)"
C_TEXT   = "color(251)"
C_BRIGHT = "white"
C_BRAND  = "bold bright_green"


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_serve_analysis(db_path: str) -> dict[str, str]:
    with sqlite3.connect(db_path) as con:
        cur = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='serve_analysis'"
        )
        if cur.fetchone() is None:
            return {}
        rows = con.execute("SELECT key, value FROM serve_analysis").fetchall()
        return {k: v for k, v in rows}


def _load_metrics_timeseries(db_path: str, metric_like: str) -> list[tuple[float, float]]:
    """Return [(sample_time, value)] for the first metric matching `metric_like`."""
    try:
        with sqlite3.connect(db_path) as con:
            rows = con.execute(
                "SELECT sample_time, value FROM vllm_metrics "
                "WHERE metric LIKE ? ORDER BY sample_time ASC",
                (f"%{metric_like}%",),
            ).fetchall()
            return [(float(t), float(v)) for t, v in rows]
    except Exception:
        return []


# ── value helpers ─────────────────────────────────────────────────────────────

def _f(kv: dict[str, str], key: str, default: float = -1.0) -> float:
    try:
        return float(kv.get(key, "")) or default
    except (ValueError, TypeError):
        return default


def _i(kv: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(kv.get(key, "") or default)
    except (ValueError, TypeError):
        return default


def _s(kv: dict[str, str], key: str, default: str = "") -> str:
    return kv.get(key, default)


def _bool(kv: dict[str, str], key: str) -> bool:
    return kv.get(key) == "true"


def _ms(v: float) -> str:
    if v < 0:
        return "—"
    if v < 1:
        return f"{v * 1000:.1f} µs"
    return f"{v:.1f} ms"


def _pct(v: float) -> str:
    return "—" if v < 0 else f"{v:.1f}%"


def _latency_color(ms: float, warn: float = 200.0, crit: float = 500.0) -> str:
    if ms < 0:
        return C_MUTED
    if ms >= crit:
        return C_RED
    if ms >= warn:
        return C_AMBER
    return C_TEXT


# ── bar drawing ───────────────────────────────────────────────────────────────

_BAR_FULL = "█"
_BAR_EMPTY = "░"
_BAR_WIDTH = 32


def _make_bar(pct: float, width: int = _BAR_WIDTH, color: str = C_GREEN) -> Text:
    filled = round(pct / 100 * width)
    bar = Text()
    bar.append(_BAR_FULL * filled, style=color)
    bar.append(_BAR_EMPTY * (width - filled), style=C_DIM)
    return bar


# ── plotext chart builder ─────────────────────────────────────────────────────

def _build_chart(
    times: list[float],
    values: list[float],
    label: str,
    unit: str,
    color: str,
    width: int = 46,
    height: int = 12,
) -> str:
    if not times or not values:
        return f"  {label}\n  (no data)\n"
    plt.clf()
    plt.theme("dark")
    plt.plotsize(width, height)
    plt.canvas_color("default")
    plt.axes_color("default")
    plt.ticks_color("default")
    plt.plot(times, values, color=color, label=label)
    plt.xlabel("seconds")
    plt.ylabel(unit)
    try:
        return plt.build()
    except Exception:
        return f"  {label}\n  (chart error)\n"


def _build_step_chart(
    times: list[float],
    values: list[float],
    label: str,
    unit: str,
    color: str,
    width: int = 46,
    height: int = 12,
) -> str:
    if not times or not values:
        return f"  {label}\n  (no data)\n"
    # Flatten to step-function pairs
    st: list[float] = []
    sv: list[float] = []
    for i, (t, v) in enumerate(zip(times, values)):
        st.append(t)
        sv.append(v)
        if i < len(times) - 1:
            st.append(times[i + 1] - 0.001)
            sv.append(v)
    plt.clf()
    plt.theme("dark")
    plt.plotsize(width, height)
    plt.canvas_color("default")
    plt.axes_color("default")
    plt.ticks_color("default")
    plt.plot(st, sv, color=color, label=label)
    plt.xlabel("seconds")
    plt.ylabel(unit)
    try:
        return plt.build()
    except Exception:
        return f"  {label}\n  (chart error)\n"


# ── section header ────────────────────────────────────────────────────────────

def _section(console: "Console", title: str) -> None:
    rule = Text()
    rule.append(f" {title} ", style=f"bold {C_MUTED}")
    console.print()
    console.rule(rule, style=C_DIM, align="left", characters="─")


# ── main renderer ─────────────────────────────────────────────────────────────

def render(db_path: str) -> int:
    """Render a rich serve-report to stdout. Returns exit code."""
    if not DEPS_AVAILABLE:
        return 1  # caller falls back to C++ binary

    p = Path(db_path)
    if not p.exists():
        print(f"error: file not found: {db_path}", file=sys.stderr)
        return 1

    kv = _load_serve_analysis(db_path)
    if not kv:
        print(f"error: no serve analysis data in {db_path}", file=sys.stderr)
        print("Run 'hotpath serve-profile' first.", file=sys.stderr)
        return 1

    console = Console(highlight=False)

    # ── header ──────────────────────────────────────────────────────────────
    console.print()
    header = Text()
    header.append("hotpath", style=f"bold {C_GREEN}")
    header.append("  serve-report", style=C_BODY)
    header.append(f"  {db_path}", style=C_MUTED)
    console.print(header)

    model   = _s(kv, "meta.model", "unknown model")
    engine  = _s(kv, "meta.engine", "vLLM")
    gpu_cnt = _i(kv, "meta.gpu_count", 1)
    gpu_nm  = _s(kv, "meta.gpu_name", "GPU")
    gpu_str = f"{gpu_cnt}x {gpu_nm}" if gpu_nm else "GPU"

    meta_line = Text()
    meta_line.append(model, style=f"bold {C_BRIGHT}")
    meta_line.append("  ·  ", style=C_DIM)
    meta_line.append(engine, style=C_BODY)
    meta_line.append("  ·  ", style=C_DIM)
    meta_line.append(gpu_str, style=C_CYAN)
    console.print(meta_line)

    flags: list[str] = []
    if _bool(kv, "latency.server_timing_available"):
        flags.append("server-log")
    if _bool(kv, "prefix.available"):
        flags.append("prefix-caching")
    if flags:
        flag_text = Text()
        for f in flags:
            flag_text.append(f"[{f}]", style=C_MUTED)
            flag_text.append(" ", style="")
        console.print(flag_text)

    console.rule(style=C_DIM)

    # ── summary cards ────────────────────────────────────────────────────────
    total_req  = _i(kv, "meta.total_requests")
    duration   = _f(kv, "meta.duration_seconds")
    rps        = _f(kv, "meta.throughput_rps")
    tok_s      = _f(kv, "meta.token_throughput", -1.0)

    cards = Table.grid(expand=True, padding=(0, 3))
    cards.add_column(justify="left", ratio=1)
    cards.add_column(justify="left", ratio=1)
    cards.add_column(justify="left", ratio=1)
    cards.add_column(justify="left", ratio=1)

    def _card(label: str, val: str, unit: str) -> Text:
        t = Text()
        t.append(f"{val}", style=f"bold {C_BRIGHT}")
        t.append(f"  {label}", style=C_MUTED)
        t.append(f"\n{unit}", style=C_DIM)
        return t

    rps_str  = f"{rps:.1f}"  if rps >= 0 else "—"
    toks_str = f"{tok_s:,.0f}" if tok_s >= 0 else "—"
    dur_str  = f"{duration:.1f}s" if duration >= 0 else "—"

    cards.add_row(
        _card("requests", f"{total_req:,}", "completed"),
        _card("duration", dur_str, ""),
        _card("req/s", rps_str, "throughput"),
        _card("tok/s", toks_str, "token rate") if tok_s >= 0 else _card("", "", ""),
    )
    console.print(Padding(cards, (1, 0)))
    console.rule(style=C_DIM)

    # ── latency table ────────────────────────────────────────────────────────
    _section(console, "LATENCY")

    lat = Table(
        show_header=True,
        header_style=C_MUTED,
        box=None,
        padding=(0, 2),
        expand=False,
    )
    lat.add_column("", style=C_BODY, no_wrap=True, min_width=22)
    lat.add_column("p50", justify="right", no_wrap=True, min_width=10)
    lat.add_column("p90", justify="right", no_wrap=True, min_width=10)
    lat.add_column("p99", justify="right", no_wrap=True, min_width=10)

    queue_avail  = _bool(kv, "latency.queue_available")
    server_avail = _bool(kv, "latency.server_timing_available")

    def _lat_row(
        label: str,
        p50: float,
        p90: float,
        p99: float,
        emphasis: bool = False,
        warn_p99: float = 200.0,
        crit_p99: float = 600.0,
    ) -> None:
        if p50 < 0 and p90 < 0 and p99 < 0:
            return
        lbl_style = f"bold {C_TEXT}" if emphasis else C_BODY
        prefix = Text()
        if emphasis:
            prefix.append("→ ", style=C_GREEN)
            prefix.append(label, style=f"bold {C_TEXT}")
        else:
            prefix = Text(label, style=lbl_style)

        p50_t = Text(_ms(p50), style=C_TEXT if p50 < 0 else (C_GREEN if p50 < 50 else C_TEXT))
        p90_t = Text(_ms(p90), style=C_TEXT)
        p99_t = Text(_ms(p99), style=_latency_color(p99, warn_p99, crit_p99))

        lat.add_row(prefix, p50_t, p90_t, p99_t)

    if queue_avail:
        _lat_row("Queue wait",
                 _f(kv, "latency.queue_p50"),
                 _f(kv, "latency.queue_p90"),
                 _f(kv, "latency.queue_p99"),
                 warn_p99=20, crit_p99=50)

    if server_avail:
        _lat_row("Prefill (server)",
                 _f(kv, "latency.server_prefill_p50"),
                 _f(kv, "latency.server_prefill_p90"),
                 _f(kv, "latency.server_prefill_p99"),
                 warn_p99=50, crit_p99=150)
        _lat_row("Decode (server)",
                 _f(kv, "latency.server_decode_p50"),
                 _f(kv, "latency.server_decode_p90"),
                 _f(kv, "latency.server_decode_p99"),
                 warn_p99=300, crit_p99=700)
        _lat_row("Decode (per-tok)",
                 _f(kv, "latency.decode_per_token_p50"),
                 _f(kv, "latency.decode_per_token_p90"),
                 _f(kv, "latency.decode_per_token_p99"),
                 warn_p99=10, crit_p99=20)
    else:
        _lat_row("Prefill",
                 _f(kv, "latency.prefill_p50"),
                 _f(kv, "latency.prefill_p90"),
                 _f(kv, "latency.prefill_p99"),
                 warn_p99=50, crit_p99=150)
        _lat_row("Decode",
                 _f(kv, "latency.decode_total_p50"),
                 _f(kv, "latency.decode_total_p90"),
                 _f(kv, "latency.decode_total_p99"),
                 warn_p99=300, crit_p99=700)

    _lat_row("End-to-end",
             _f(kv, "latency.e2e_p50"),
             _f(kv, "latency.e2e_p90"),
             _f(kv, "latency.e2e_p99"),
             emphasis=True,
             warn_p99=400, crit_p99=800)

    console.print(Padding(lat, (0, 0, 0, 2)))

    # ── GPU phase ────────────────────────────────────────────────────────────
    phase_avail = _bool(kv, "phase.available")
    if phase_avail:
        _section(console, "GPU PHASE")
        prefill_pct  = _f(kv, "phase.prefill_pct", 0)
        decode_pct   = _f(kv, "phase.decode_pct", 0)
        other_pct    = _f(kv, "phase.other_pct", 0)
        schedule_pct = max(0.0, 100.0 - prefill_pct - decode_pct - other_pct)

        phases = [
            ("Prefill",  prefill_pct,  C_PURPLE),
            ("Decode",   decode_pct,   C_GREEN),
            ("Schedule", schedule_pct, C_AMBER),
            ("Idle",     other_pct,    C_MUTED),
        ]
        phase_table = Table.grid(padding=(0, 2))
        phase_table.add_column(min_width=10, style=C_BODY)
        phase_table.add_column(min_width=6,  justify="right")
        phase_table.add_column()

        for name, pct, color in phases:
            pct_text = Text(f"{pct:.1f}%", style=f"bold {color}")
            bar = _make_bar(pct, color=color)
            phase_table.add_row(Text(name, style=C_BODY), pct_text, bar)

        console.print(Padding(phase_table, (0, 0, 0, 2)))

    # ── throughput + batch charts ────────────────────────────────────────────
    tput_series  = _load_metrics_timeseries(db_path, "generation_throughput")
    batch_series = _load_metrics_timeseries(db_path, "num_requests_running")

    if tput_series or batch_series:
        _section(console, "THROUGHPUT / BATCH SIZE")

        chart_width  = 50
        chart_height = 12

        if tput_series:
            t_times  = [r[0] for r in tput_series]
            t_values = [r[1] for r in tput_series]
            # Normalise times to start at 0
            t0 = t_times[0]
            t_times = [t - t0 for t in t_times]
            tput_chart = _build_chart(
                t_times, t_values,
                "Throughput", "tok/s", "green",
                width=chart_width, height=chart_height,
            )
        else:
            tput_chart = "  (no throughput data)\n"

        if batch_series:
            b_times  = [r[0] for r in batch_series]
            b_values = [r[1] for r in batch_series]
            b0 = b_times[0]
            b_times = [t - b0 for t in b_times]
            batch_chart = _build_step_chart(
                b_times, b_values,
                "Batch Size", "requests", "orange",
                width=chart_width, height=chart_height,
            )
        else:
            batch_chart = "  (no batch size data)\n"

        # side-by-side via Columns — use Text.from_ansi so Rich handles the
        # escape codes that plotext embeds rather than printing them raw
        from rich.panel import Panel
        from rich.text import Text as RichText

        def _chart_panel(raw: str, title: str) -> Panel:
            return Panel(
                RichText.from_ansi(raw.rstrip()),
                title=title,
                border_style=C_DIM,
                title_align="left",
                padding=(0, 1),
            )

        cols = Columns(
            [
                _chart_panel(tput_chart,  "Throughput tok/s"),
                _chart_panel(batch_chart, "Batch Size requests"),
            ],
            equal=True,
        )
        console.print(Padding(cols, (0, 0, 0, 0)))

    # ── KV cache + prefix sharing ────────────────────────────────────────────
    cache_avail  = _bool(kv, "cache.hit_rate_available")
    prefix_avail = _bool(kv, "prefix.available")

    if cache_avail or prefix_avail:
        _section(console, "KV CACHE  ·  PREFIX SHARING")

        two_col = Table.grid(expand=True, padding=(0, 4))
        two_col.add_column(ratio=1)
        two_col.add_column(ratio=1)

        def _kv_block(pairs: list[tuple[str, str, str]]) -> Table:
            t = Table.grid(padding=(0, 2))
            t.add_column(style=C_BODY, min_width=20)
            t.add_column(justify="right")
            for label, val, color in pairs:
                t.add_row(label, Text(val, style=f"bold {color}"))
            return t

        cache_pairs: list[tuple[str, str, str]] = []
        if cache_avail:
            # cache.hit_rate is stored as 0–1 fraction
            hit = _f(kv, "cache.hit_rate", 0) * 100
            color = C_GREEN if hit > 60 else (C_AMBER if hit > 30 else C_RED)
            cache_pairs.append(("Hit rate",    f"{hit:.1f}%",  color))
        if _bool(kv, "cache.usage_available"):
            avg_u  = _f(kv, "cache.avg_usage",  0) * 100
            peak_u = _f(kv, "cache.peak_usage", 0) * 100
            cache_pairs.append(("Avg usage",  f"{avg_u:.1f}%",  C_TEXT))
            cache_pairs.append(("Peak usage", f"{peak_u:.1f}%", C_AMBER if peak_u > 90 else C_TEXT))
            evict = _i(kv, "cache.evictions")
            cache_pairs.append(("Evictions",  str(evict), C_AMBER if evict > 5 else C_TEXT))

        prefix_pairs: list[tuple[str, str, str]] = []
        if prefix_avail:
            unique_p = _i(kv, "prefix.unique_prefixes")
            # cacheable_tokens_pct is already in 0–100 scale
            cache_tok_pct = _f(kv, "prefix.cacheable_tokens_pct", 0)
            prefix_pairs.append(("Unique prefixes",  str(unique_p), C_TEXT))
            prefix_pairs.append(("Cacheable tokens", f"{cache_tok_pct:.1f}%",
                                  C_GREEN if cache_tok_pct > 50 else C_AMBER))
            if cache_avail:
                actual_hr = _f(kv, "cache.hit_rate", 0) * 100
                prefix_pairs.append(("Actual hit rate", f"{actual_hr:.1f}%",
                                      C_GREEN if actual_hr > 60 else C_AMBER))

        two_col.add_row(
            Padding(_kv_block(cache_pairs),   (0, 0, 0, 2)) if cache_pairs else Text(""),
            Padding(_kv_block(prefix_pairs),  (0, 0, 0, 0)) if prefix_pairs else Text(""),
        )
        console.print(two_col)

    # ── cache hit distribution ───────────────────────────────────────────────
    hist_avail = _bool(kv, "cache.histogram_available")
    if hist_avail:
        _section(console, "CACHE HIT DISTRIBUTION")

        buckets = ["0%", "1–25%", "25–50%", "50–75%", "75%+"]
        counts  = [_i(kv, f"cache.histogram_{i}") for i in range(5)]
        total_h = sum(counts) or 1
        max_c   = max(counts) or 1

        hist_tbl = Table.grid(padding=(0, 1))
        hist_tbl.add_column(min_width=7,  justify="right", style=C_MUTED)
        hist_tbl.add_column(min_width=30)
        hist_tbl.add_column(min_width=14, justify="right", style=C_MUTED)

        for i, (bucket, cnt) in enumerate(zip(buckets, counts)):
            pct = cnt / total_h * 100
            bar_color = C_GREEN if i >= 3 else (C_AMBER if i >= 1 else C_MUTED)
            bar = _make_bar(cnt / max_c * 100, width=28, color=bar_color)
            hist_tbl.add_row(
                bucket,
                bar,
                f"{cnt:,}  ({pct:.1f}%)",
            )

        console.print(Padding(hist_tbl, (0, 0, 0, 2)))

    # ── disaggregation advisor ───────────────────────────────────────────────
    _section(console, "DISAGGREGATION ADVISOR")

    workload_cls = _s(kv, "disagg.workload_class", "")
    caveat       = _s(kv, "disagg.caveat", "")
    should_disagg = _bool(kv, "disagg.should")

    adv_tbl = Table.grid(padding=(0, 2))
    adv_tbl.add_column(style=C_MUTED, min_width=22)
    adv_tbl.add_column()

    if workload_cls:
        adv_tbl.add_row("Workload class", Text(workload_cls, style=f"bold {C_TEXT}"))

    # Prefill contention
    mono_ttft  = _f(kv, "disagg.mono_p99_ttft", -1)
    disagg_ttft = _f(kv, "disagg.disagg_p99_ttft", -1)
    if mono_ttft > 0 and disagg_ttft > 0:
        improvement = (mono_ttft - disagg_ttft) / mono_ttft * 100
        adv_tbl.add_row(
            "Prefill contention",
            Text(f"adds {improvement:.0f}% to p99 decode latency", style=C_AMBER),
        )

    if caveat:
        adv_tbl.add_row("Note", Text(caveat, style=C_MUTED))

    console.print(Padding(adv_tbl, (0, 0, 0, 2)))

    # Recommendation badge
    console.print()
    rec_line = Text()
    rec_line.append("  Recommendation  ", style=C_MUTED)
    if should_disagg:
        rec_line.append(" ✦ DISAGGREGATE ", style=f"bold reverse {C_GREEN}")
    else:
        rec_line.append(" ✓ KEEP MONOLITHIC ", style=f"bold reverse {C_AMBER}")
    console.print(rec_line)
    console.print()

    # Projection table
    proj_rps  = _f(kv, "disagg.disagg_throughput", -1)
    tput_impr = _f(kv, "disagg.throughput_improvement", -1)
    mono_ttft  = _f(kv, "disagg.mono_p99_ttft", -1)
    disagg_ttft = _f(kv, "disagg.disagg_p99_ttft", -1)

    if rps > 0 and proj_rps > 0:
        proj_tbl = Table(
            show_header=True,
            header_style=C_MUTED,
            box=None,
            padding=(0, 2),
        )
        proj_tbl.add_column("",           style=C_BODY, min_width=22)
        proj_tbl.add_column("Current",    justify="right", min_width=10)
        proj_tbl.add_column("Projected",  justify="right", min_width=10)
        proj_tbl.add_column("Improvement", justify="right", min_width=12)

        def _impr(cur: float, proj: float, higher_better: bool = True) -> Text:
            if cur <= 0 or proj <= 0:
                return Text("—", style=C_MUTED)
            delta_pct = (proj - cur) / abs(cur) * 100
            good = (delta_pct > 0) == higher_better
            sign  = "+" if delta_pct > 0 else ""
            color = C_GREEN if good else C_RED
            return Text(f"{sign}{delta_pct:.0f}%", style=f"bold {color}")

        tput_pct_impr = (tput_impr - 1.0) * 100 if tput_impr > 0 else 0.0

        proj_tbl.add_row(
            "Throughput (req/s)",
            f"{rps:.1f}",
            f"{proj_rps:.1f}",
            Text(f"+{tput_pct_impr:.0f}%", style=f"bold {C_GREEN}") if tput_pct_impr > 0 else Text("—", style=C_MUTED),
        )
        if mono_ttft > 0 and disagg_ttft > 0:
            proj_tbl.add_row(
                "p99 TTFT (ms)",
                f"{mono_ttft:.1f}",
                f"{disagg_ttft:.1f}",
                _impr(mono_ttft, disagg_ttft, higher_better=False),
            )

        console.print(Padding(proj_tbl, (0, 0, 0, 2)))

    # Advisor notes
    opt_p      = _i(kv, "disagg.optimal_p")
    opt_d      = _i(kv, "disagg.optimal_d")
    min_bw     = _f(kv, "disagg.min_bandwidth", -1)

    notes = Table.grid(padding=(0, 2))
    notes.add_column(style=C_MUTED, min_width=24)
    notes.add_column()

    if opt_p > 0 and opt_d > 0:
        notes.add_row(
            "Optimal P/D ratio",
            Text(f"1:{opt_d // opt_p}  ({opt_p} prefill, {opt_d} decode)", style=C_TEXT),
        )
    if min_bw > 0:
        warn = min_bw < 50
        notes.add_row(
            "Min network bandwidth",
            Text(
                f"{min_bw:.0f} Gbps" + ("  (below this, disagg hurts)" if warn else ""),
                style=C_AMBER if warn else C_TEXT,
            ),
        )

    if opt_p > 0 or min_bw > 0:
        console.print()
        console.print(Padding(notes, (0, 0, 0, 2)))

    # ── footer ───────────────────────────────────────────────────────────────
    console.print()
    console.rule(style=C_DIM)
    footer = Text()
    footer.append("Run ", style=C_MUTED)
    footer.append(f"hotpath disagg-config {db_path}", style=C_BODY)
    footer.append(" for vLLM / llm-d / Dynamo config", style=C_MUTED)
    console.print(Padding(footer, (0, 0, 1, 0)))

    return 0

if __name__ == "__main__":
    if not DEPS_AVAILABLE:
        sys.exit(2)
    if len(sys.argv) < 2:
        print("usage: python3 -m hotpath_py.serve_report <serve_profile.db>", file=sys.stderr)
        sys.exit(1)
    sys.exit(render(sys.argv[1]))
