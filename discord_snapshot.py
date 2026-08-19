"""
discord_snapshot.py
--------------------
Fetches the same Trend Analyzer metrics used in `trend_analyzer.py`
(Price, YTD %, 5-Day %, 50-DMA %, Trend, and a Bespoke-style 50-DMA
standard-deviation Status / Timing), renders them as a dark-themed
table image with matplotlib, and posts that image to a Discord
channel via an incoming webhook.

Intended to run headless (e.g. from a GitHub Actions workflow).

Environment variables:
    DISCORD_WEBHOOK_URL - required. The Discord webhook URL to post to.

Usage:
    python discord_snapshot.py
"""

import datetime as dt
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # headless backend - no display needed in CI
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# --------------------------------------------------------------------------
# Config - kept in sync with trend_analyzer.py
# --------------------------------------------------------------------------
INDEX_TICKERS = ["SPY", "QQQ", "RSP", "IWM", "VOO"]
SECTOR_ETF_TICKERS = [
    "XLE", "XLI", "XLF", "XLV", "XLP",
    "XLC", "XLB", "XLK", "XLY", "XLRE", "XLU",
    "IGV", "SOXX",
]
TICKERS = INDEX_TICKERS + SECTOR_ETF_TICKERS

OUTPUT_IMAGE_PATH = "table_snapshot.png"

# --------------------------------------------------------------------------
# Yahoo Finance session (same hardening as trend_analyzer.py, to reduce
# connection failures from CI / cloud runners)
# --------------------------------------------------------------------------
def build_yf_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


YF_SESSION = build_yf_session()


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a yfinance history DataFrame to plain, single-level columns."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    field_level = None
    for level in range(df.columns.nlevels):
        values = set(df.columns.get_level_values(level))
        if {"Close", "Open", "High", "Low", "Adj Close"} & values:
            field_level = level
            break

    flat = df.copy()
    flat.columns = flat.columns.get_level_values(
        field_level if field_level is not None else 0
    )
    flat = flat.loc[:, ~flat.columns.duplicated()]
    return flat


def safe_get_column(df: pd.DataFrame, primary: str, fallback: str = None) -> pd.Series:
    """Safely extract a column as a Series, with an optional fallback column."""
    for name in (primary, fallback):
        if name and name in df.columns:
            col = df[name]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            return col.dropna()
    raise KeyError(f"Neither '{primary}' nor '{fallback}' found in downloaded data.")


def fetch_history_with_retry(ticker: str, retries: int = 3, delay: float = 1.5) -> pd.DataFrame:
    """Fetch daily history for a single ticker with a small retry/backoff loop."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            tk = yf.Ticker(ticker, session=YF_SESSION)
            hist = tk.history(
                period="18mo",
                interval="1d",
                auto_adjust=False,
                actions=False,
                timeout=15,
            )
            if hist is not None and not hist.empty:
                return hist
            last_exc = ValueError("Empty response from Yahoo Finance.")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

        if attempt < retries:
            time.sleep(delay * attempt)

    raise last_exc if last_exc else RuntimeError("Unknown error fetching history.")


# --------------------------------------------------------------------------
# Status / Timing logic - identical to trend_analyzer.py
# --------------------------------------------------------------------------
def zscore_status(z: float) -> str:
    """
    Bespoke-style 50-DMA standard-deviation band classification:
      Z = (Price - SMA_50) / STD_50
        Z >= 2.0            -> Extreme OB
        1.0 <= Z < 2.0       -> Overbought
        -1.0 < Z < 1.0       -> Neutral
        -2.0 < Z <= -1.0     -> Oversold
        Z <= -2.0            -> Extreme OS
    """
    if pd.isna(z):
        return "N/A"
    if z >= 2.0:
        return "Extreme OB"
    if z >= 1.0:
        return "Overbought"
    if z <= -2.0:
        return "Extreme OS"
    if z <= -1.0:
        return "Oversold"
    return "Neutral"


def timing_from_status(status: str) -> str:
    mapping = {
        "Extreme OB": "Poor",
        "Overbought": "Poor",
        "Neutral": "Neutral",
        "Oversold": "Good",
        "Extreme OS": "Good",
        "N/A": "N/A",
    }
    return mapping.get(status, "N/A")


def ticker_category(ticker: str) -> str:
    if ticker in INDEX_TICKERS:
        return "Index"
    if ticker in SECTOR_ETF_TICKERS:
        return "Sector ETF"
    return "Other"


# --------------------------------------------------------------------------
# Metric computation
# --------------------------------------------------------------------------
def compute_ticker_metrics(ticker: str) -> dict:
    """
    Compute Price, YTD %, 5-Day %, 50-DMA %, Trend, Status (Z-score based),
    and Timing for a single ticker - same logic as trend_analyzer.py.
    """
    row = {
        "Ticker": ticker,
        "Category": ticker_category(ticker),
        "Price": np.nan,
        "YTD %": np.nan,
        "5-Day %": np.nan,
        "50-DMA %": np.nan,
        "Trend": "N/A",
        "Status": "N/A",
        "Timing": "N/A",
        "error": None,
    }
    try:
        hist = fetch_history_with_retry(ticker)
        hist = flatten_columns(hist)

        close = safe_get_column(hist, "Close", fallback="Adj Close")
        if close.empty:
            row["error"] = "No valid close prices found."
            return row

        price = float(close.iloc[-1])

        # 5-Day % change
        if len(close) >= 6:
            five_day_pct = (price / float(close.iloc[-6]) - 1) * 100
        elif len(close) >= 2:
            five_day_pct = (price / float(close.iloc[0]) - 1) * 100
        else:
            five_day_pct = np.nan

        # YTD % change
        current_year = dt.datetime.now().year
        ytd_hist = close[close.index.year == current_year]
        if not ytd_hist.empty:
            ytd_pct = (price / float(ytd_hist.iloc[0]) - 1) * 100
        else:
            ytd_pct = np.nan

        # 50-Day Moving Average & Standard Deviation
        if len(close) >= 50:
            sma50 = float(close.rolling(window=50).mean().iloc[-1])
            std50 = float(close.rolling(window=50).std().iloc[-1])
            dma50_pct = (price / sma50 - 1) * 100
        else:
            sma50 = np.nan
            std50 = np.nan
            dma50_pct = np.nan

        # 50-DMA Standard-Deviation Z-Score (Bespoke-style)
        if not pd.isna(sma50) and not pd.isna(std50) and std50 > 0:
            zscore = (price - sma50) / std50
        else:
            zscore = np.nan

        status = zscore_status(zscore)
        timing = timing_from_status(status)
        trend = "Bullish" if (not pd.isna(dma50_pct) and dma50_pct >= 0) else (
            "Bearish" if not pd.isna(dma50_pct) else "N/A"
        )

        row.update(
            {
                "Price": round(price, 2),
                "YTD %": round(ytd_pct, 2) if not pd.isna(ytd_pct) else np.nan,
                "5-Day %": round(five_day_pct, 2) if not pd.isna(five_day_pct) else np.nan,
                "50-DMA %": round(dma50_pct, 2) if not pd.isna(dma50_pct) else np.nan,
                "Trend": trend,
                "Status": status,
                "Timing": timing,
            }
        )
        return row

    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row


def build_dataframe(tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    errors = []
    for t in tickers:
        r = compute_ticker_metrics(t)
        if r.get("error"):
            errors.append(f"{t}: {r['error']}")
        rows.append(r)

    df = pd.DataFrame(rows).drop(columns=["error"]).set_index("Ticker")
    return df, errors


# --------------------------------------------------------------------------
# Color scheme - identical hex values to trend_analyzer.py, for a
# consistent look between the Streamlit app and the Discord snapshot.
# --------------------------------------------------------------------------
STATUS_STYLE = {
    "Extreme OB": ("#ff4c4c", "#ffffff"),
    "Overbought": ("#ffb3b3", "#000000"),
    "Neutral":    ("#ffffff", "#000000"),
    "Oversold":   ("#b3ffb3", "#000000"),
    "Extreme OS": ("#33cc33", "#ffffff"),
    "N/A":        ("#3a3a3a", "#bbbbbb"),
}

TIMING_STYLE = {
    "Good":    ("#33cc33", "#ffffff"),
    "Neutral": ("#ffb347", "#000000"),
    "Poor":    ("#ff4c4c", "#ffffff"),
    "N/A":     ("#3a3a3a", "#bbbbbb"),
}

TREND_ARROWS = {"Bullish": "▲", "Bearish": "▼", "N/A": "–"}
TREND_TEXT_COLORS = {"Bullish": "#2ecc71", "Bearish": "#ff4c4c", "N/A": "#999999"}

# Dark-theme canvas colors
FIG_BG = "#0e0e10"
CELL_BG = "#1b1b1f"
HEADER_BG = "#111113"
GRID_COLOR = "#3a3a3a"
TEXT_COLOR = "#f5f5f5"

NEG_RGB = (255, 76, 76)   # red   - matches STATUS_STYLE "Extreme OB"
ZERO_RGB = (58, 58, 58)   # dark gray - blends with the dark theme
POS_RGB = (51, 204, 51)   # green - matches STATUS_STYLE "Extreme OS"


def _lerp(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def pct_bg_rgb(val, vmin, vmax):
    """Red -> dark gray -> green gradient, same spirit as trend_analyzer.py."""
    if pd.isna(val):
        return ZERO_RGB
    v = max(min(val, vmax), vmin)
    if v >= 0:
        t = v / vmax if vmax else 0
        return _lerp(ZERO_RGB, POS_RGB, t)
    t = v / vmin if vmin else 0  # v negative / vmin negative -> positive fraction
    return _lerp(ZERO_RGB, NEG_RGB, t)


def rgb_to_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[max(0, min(255, int(c))) for c in rgb])


def text_color_for_rgb(rgb) -> str:
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#111111" if luminance > 150 else "#ffffff"


# --------------------------------------------------------------------------
# Image rendering
# --------------------------------------------------------------------------
def render_table_image(df: pd.DataFrame, output_path: str = OUTPUT_IMAGE_PATH) -> str:
    # Note: the descriptive Hebrew "Name" column from trend_analyzer.py is
    # intentionally omitted here. matplotlib has no bidi/RTL text shaping
    # engine, so right-to-left strings render in the wrong visual order.
    # Ticker symbols and English labels are unaffected.
    columns = ["Ticker", "Category", "Price", "YTD %", "5-Day %", "50-DMA %", "Trend", "Status", "Timing"]
    n_cols = len(columns)

    cell_text = []
    cell_bg_colors = []
    cell_text_colors = []

    for ticker, row in df.iterrows():
        vals, bgs, fgs = [], [], []

        # Ticker
        vals.append(ticker)
        bgs.append(CELL_BG)
        fgs.append(TEXT_COLOR)

        # Category
        vals.append(row["Category"])
        bgs.append(CELL_BG)
        fgs.append("#bbbbbb")

        # Price
        price = row["Price"]
        vals.append(f"${price:,.2f}" if pd.notna(price) else "N/A")
        bgs.append(CELL_BG)
        fgs.append(TEXT_COLOR)

        # YTD %, 5-Day %, 50-DMA % (gradient-colored)
        for col, (vmin, vmax) in (
            ("YTD %", (-20, 20)),
            ("5-Day %", (-5, 5)),
            ("50-DMA %", (-10, 10)),
        ):
            v = row[col]
            rgb = pct_bg_rgb(v, vmin, vmax)
            vals.append(f"{v:+.2f}%" if pd.notna(v) else "N/A")
            bgs.append(rgb_to_hex(rgb))
            fgs.append(text_color_for_rgb(rgb))

        # Trend (arrow icon only, colored text, no background fill)
        trend = row["Trend"]
        vals.append(TREND_ARROWS.get(trend, "–"))
        bgs.append(CELL_BG)
        fgs.append(TREND_TEXT_COLORS.get(trend, "#999999"))

        # Status
        status = row["Status"]
        bg_hex, fg_hex = STATUS_STYLE.get(status, STATUS_STYLE["N/A"])
        vals.append(status)
        bgs.append(bg_hex)
        fgs.append(fg_hex)

        # Timing
        timing = row["Timing"]
        bg_hex, fg_hex = TIMING_STYLE.get(timing, TIMING_STYLE["N/A"])
        vals.append(timing)
        bgs.append(bg_hex)
        fgs.append(fg_hex)

        cell_text.append(vals)
        cell_bg_colors.append(bgs)
        cell_text_colors.append(fgs)

    n_rows = len(df)
    fig_width = 14
    fig_height = 0.5 * (n_rows + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(FIG_BG)
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellColours=cell_bg_colors,
        colColours=[HEADER_BG] * n_cols,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    bold_cols = {columns.index(c) for c in ("Trend", "Status", "Timing")}
    for (r_idx, c_idx), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.set_linewidth(0.6)
        if r_idx == 0:
            cell.get_text().set_color("#ffffff")
            cell.get_text().set_fontweight("bold")
        else:
            data_row = r_idx - 1
            cell.get_text().set_color(cell_text_colors[data_row][c_idx])
            if c_idx in bold_cols:
                cell.get_text().set_fontweight("bold")

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ax.set_title(
        f"Trend Analyzer — Daily Snapshot ({timestamp})",
        color=TEXT_COLOR,
        fontsize=14,
        fontweight="bold",
        pad=16,
    )

    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return output_path


# --------------------------------------------------------------------------
# Discord webhook
# --------------------------------------------------------------------------
def send_to_discord(image_path: str, errors: list[str] = None) -> None:
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("ERROR: DISCORD_WEBHOOK_URL environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    message = f"📊 **Trend Analyzer — Daily Snapshot** ({timestamp})"
    if errors:
        shown = "; ".join(errors[:5])
        more = "" if len(errors) <= 5 else f" (+{len(errors) - 5} more)"
        message += f"\n⚠️ Issues fetching some tickers: {shown}{more}"

    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/png")}
            data = {"content": message}
            resp = requests.post(webhook_url, data=data, files=files, timeout=30)
    except requests.RequestException as exc:
        print(f"ERROR: Failed to reach Discord webhook: {exc}", file=sys.stderr)
        sys.exit(1)

    if resp.status_code not in (200, 204):
        print(
            f"ERROR: Discord webhook returned status {resp.status_code}: {resp.text}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Successfully posted snapshot to Discord (status {resp.status_code}).")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    print(f"Fetching data for {len(TICKERS)} tickers...")
    df, errors = build_dataframe(TICKERS)

    if errors:
        print("Warnings while fetching data:")
        for e in errors:
            print(" -", e)

    if df["Price"].isna().all():
        print("ERROR: No data could be retrieved for any ticker. Aborting.", file=sys.stderr)
        sys.exit(1)

    image_path = render_table_image(df)
    print(f"Saved table image to: {image_path}")

    send_to_discord(image_path, errors=errors)


if __name__ == "__main__":
    main()
