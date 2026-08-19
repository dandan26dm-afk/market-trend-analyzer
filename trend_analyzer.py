"""
Trend Analyzer - Streamlit App
--------------------------------
Tracks SPY, QQQ, RSP, IWM and shows price trend, YTD / 5-day / 50-DMA
performance, RSI(14) based overbought/oversold status, and a 52-week
trading-range indicator - all in a color-coded (heatmap-style) table.

Deploy on Streamlit Cloud:
    1. Put this file in a repo as `app.py` (or `trend_analyzer.py`).
    2. Add a `requirements.txt` with:
         streamlit
         yfinance
         pandas
         numpy
    3. Point Streamlit Cloud at this file.
"""

import datetime as dt
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
INDEX_TICKERS = ["SPY", "QQQ", "RSP", "IWM", "VOO"]
SECTOR_ETF_TICKERS = [
    "XLE", "XLI", "XLF", "XLV", "XLP",
    "XLC", "XLB", "XLK", "XLY", "XLRE", "XLU",
    "IGV", "SOXX",
]
TICKERS = INDEX_TICKERS + SECTOR_ETF_TICKERS

# Human-readable name / description for each ticker, shown in the "Name" column.
TICKER_NAMES = {
    "SPY": "מדד S&P 500 (משקל שוק)",
    "QQQ": "מדד הנאסד\"ק 100",
    "RSP": "מדד S&P 500 (משקל שווה)",
    "IWM": "מדד ראסל 2000 (Small Cap)",
    "VOO": "מדד S&P 500 (Vanguard)",
    "IGV": "סקטור התוכנה",
    "SOXX": "סקטור השבבים והמוליכים למחצה",
    "XLE": "סקטור האנרגיה",
    "XLI": "סקטור התעשייה",
    "XLF": "סקטור הפיננסים",
    "XLV": "סקטור הבריאות",
    "XLP": "סקטור צריכה בסיסית",
    "XLC": "סקטור התקשורת והמדיה",
    "XLB": "סקטור חומרי גלם",
    "XLK": "סקטור הטכנולוגיה",
    "XLY": "סקטור צריכה מחזורית",
    "XLRE": "סקטור הנדל\"ן",
    "XLU": "סקטור התשתיות",
}

st.set_page_config(
    page_title="Trend Analyzer",
    page_icon="📈",
    layout="wide",
)

# --------------------------------------------------------------------------
# Yahoo Finance session
# --------------------------------------------------------------------------
# Streamlit Cloud servers sometimes get blocked / rate-limited by Yahoo
# Finance because requests arrive without a "normal" browser User-Agent.
# Building a dedicated requests.Session with a browser-like User-Agent (and
# reusing it across all tickers) significantly reduces connection failures.
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

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI (simple moving average version)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # When avg_loss is 0 and avg_gain > 0 -> RSI should be 100
    rsi = rsi.where(avg_loss != 0, 100)
    return rsi


def rsi_level(rsi_value: float) -> str:
    """
    Classify a raw RSI(14) value into the same label set used elsewhere,
    purely for coloring the RSI column itself. This is independent of the
    main 'Status' column, which is now based on 50-DMA standard-deviation
    bands (see `zscore_status` below), not RSI.
    """
    if pd.isna(rsi_value):
        return "N/A"
    if rsi_value >= 80:
        return "Extreme OB"
    if rsi_value >= 70:
        return "Overbought"
    if rsi_value <= 20:
        return "Extreme OS"
    if rsi_value <= 30:
        return "Oversold"
    return "Neutral"


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
    """
    Timing is now derived directly from the Status (Z-score) reading:
      Extreme OB / Overbought -> Poor   (stretched to the upside, caution)
      Neutral                 -> Neutral
      Oversold / Extreme OS   -> Good   (stretched to the downside, opportunity)
    """
    mapping = {
        "Extreme OB": "Poor",
        "Overbought": "Poor",
        "Neutral": "Neutral",
        "Oversold": "Good",
        "Extreme OS": "Good",
        "N/A": "N/A",
    }
    return mapping.get(status, "N/A")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a yfinance history DataFrame so it always has plain,
    single-level columns (Open/High/Low/Close/Adj Close/Volume), even if
    yfinance returned a MultiIndex (which can happen depending on the
    yfinance version / endpoint, even for a single ticker).
    """
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
    # If flattening produced duplicate column names, keep the first occurrence
    flat = flat.loc[:, ~flat.columns.duplicated()]
    return flat


def safe_get_column(df: pd.DataFrame, primary: str, fallback: str = None) -> pd.Series:
    """
    Safely extract a single column as a Series, trying `primary` first and
    then `fallback` (e.g. 'Close' -> 'Adj Close'). Raises KeyError if
    neither is available.
    """
    for name in (primary, fallback):
        if name and name in df.columns:
            col = df[name]
            if isinstance(col, pd.DataFrame):  # duplicate labels edge case
                col = col.iloc[:, 0]
            return col.dropna()
    raise KeyError(f"Neither '{primary}' nor '{fallback}' found in downloaded data.")


def fetch_history_with_retry(ticker: str, retries: int = 3, delay: float = 1.5) -> pd.DataFrame:
    """
    Fetch daily history for a single ticker using yf.Ticker(...).history(),
    with a small retry loop to smooth over transient Yahoo Finance /
    cloud-network hiccups (timeouts, empty responses, rate limits, etc.)
    """
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
            time.sleep(delay * attempt)  # simple backoff

    # All retries failed - raise the last seen exception so the caller
    # can record a clean error message for this specific ticker.
    raise last_exc if last_exc else RuntimeError("Unknown error fetching history.")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ticker_row(ticker: str) -> dict:
    """
    Pull ~18 months of daily data for a ticker and compute all the metrics.
    Each ticker is fetched independently (its own try/except) so that a
    failure on one symbol never blocks the others.
    Returns a dict with an 'error' key set if something went wrong
    (e.g. no data available, non-trading day, network/Yahoo issue, etc.)
    """
    row = {"Ticker": ticker, "error": None}
    try:
        hist = fetch_history_with_retry(ticker)
        hist = flatten_columns(hist)

        close = safe_get_column(hist, "Close", fallback="Adj Close")
        if close.empty:
            row["error"] = "No valid close prices found."
            return row

        price = float(close.iloc[-1])

        # --- 5-Day % change ---------------------------------------------
        if len(close) >= 6:
            five_day_pct = (price / float(close.iloc[-6]) - 1) * 100
        elif len(close) >= 2:
            five_day_pct = (price / float(close.iloc[0]) - 1) * 100
        else:
            five_day_pct = np.nan

        # --- YTD % change --------------------------------------------------
        current_year = dt.datetime.now().year
        ytd_hist = close[close.index.year == current_year]
        if not ytd_hist.empty:
            ytd_start = float(ytd_hist.iloc[0])
            ytd_pct = (price / ytd_start - 1) * 100
        else:
            ytd_pct = np.nan

        # --- 50-Day Moving Average & Standard Deviation ---------------------
        if len(close) >= 50:
            sma50 = float(close.rolling(window=50).mean().iloc[-1])
            std50 = float(close.rolling(window=50).std().iloc[-1])
            dma50_pct = (price / sma50 - 1) * 100
        else:
            sma50 = np.nan
            std50 = np.nan
            dma50_pct = np.nan

        # --- 50-DMA Standard-Deviation Z-Score (Bespoke-style) ---------------
        # Z = (Price - SMA_50) / STD_50
        if not pd.isna(sma50) and not pd.isna(std50) and std50 > 0:
            zscore = (price - sma50) / std50
        else:
            zscore = np.nan
        status = zscore_status(zscore)

        # --- RSI (14) --------------------------------------------------------
        # Still computed and shown as its own column, but no longer drives
        # the main 'Status' classification (that's now Z-score based above).
        rsi_series = compute_rsi(close, period=14)
        rsi_value = float(rsi_series.iloc[-1]) if not rsi_series.empty else np.nan

        # --- 52-week trading range -------------------------------------------
        # NOTE: DataFrame.last() was removed in newer pandas versions, so we
        # filter by date using the index directly instead.
        if len(hist.index) > 0:
            cutoff = hist.index.max() - pd.Timedelta(days=365)
            lookback = hist[hist.index >= cutoff]
            if lookback.empty:
                lookback = hist
        else:
            lookback = hist
        try:
            high_col = safe_get_column(lookback, "High", fallback="Close")
        except KeyError:
            high_col = close
        try:
            low_col = safe_get_column(lookback, "Low", fallback="Close")
        except KeyError:
            low_col = close
        high_52 = float(high_col.max())
        low_52 = float(low_col.min())

        if high_52 > low_52:
            position_52w = (price - low_52) / (high_52 - low_52)
            position_52w = float(np.clip(position_52w, 0, 1))
        else:
            position_52w = np.nan

        timing = timing_from_status(status)

        row.update(
            {
                "Price": round(price, 2),
                "YTD %": round(ytd_pct, 2) if not pd.isna(ytd_pct) else np.nan,
                "5-Day %": round(five_day_pct, 2) if not pd.isna(five_day_pct) else np.nan,
                "50-DMA %": round(dma50_pct, 2) if not pd.isna(dma50_pct) else np.nan,
                "Trend": "Bullish" if (not pd.isna(dma50_pct) and dma50_pct >= 0) else (
                    "Bearish" if not pd.isna(dma50_pct) else "N/A"
                ),
                "RSI (14)": round(rsi_value, 1) if not pd.isna(rsi_value) else np.nan,
                "Z-Score": round(zscore, 2) if not pd.isna(zscore) else np.nan,
                "Status": status,
                "Timing": timing,
                "52W Low": round(low_52, 2),
                "52W High": round(high_52, 2),
                "52W Position": position_52w,  # 0..1, used for progress bar
            }
        )
        return row

    except Exception as exc:  # noqa: BLE001 - want to surface any failure gracefully
        row["error"] = f"Error fetching data ({type(exc).__name__}): {exc}"
        return row


def ticker_category(ticker: str) -> str:
    if ticker in INDEX_TICKERS:
        return "Index"
    if ticker in SECTOR_ETF_TICKERS:
        return "Sector ETF"
    return "Other"


def ticker_name(ticker: str) -> str:
    return TICKER_NAMES.get(ticker, ticker)


@st.cache_data(ttl=300, show_spinner=False)
def build_dataframe(tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    errors = []
    for t in tickers:
        r = fetch_ticker_row(t)
        if r.get("error"):
            errors.append(f"**{t}**: {r['error']}")
            # Still add a placeholder row so the ticker shows up in the table
            rows.append(
                {
                    "Ticker": t,
                    "Category": ticker_category(t),
                    "Name": ticker_name(t),
                    "Price": np.nan,
                    "YTD %": np.nan,
                    "5-Day %": np.nan,
                    "50-DMA %": np.nan,
                    "Trend": "N/A",
                    "RSI (14)": np.nan,
                    "Z-Score": np.nan,
                    "Status": "N/A",
                    "Timing": "N/A",
                    "52W Low": np.nan,
                    "52W High": np.nan,
                    "52W Position": np.nan,
                }
            )
        else:
            r.pop("error", None)
            r["Category"] = ticker_category(t)
            r["Name"] = ticker_name(t)
            rows.append(r)

    df = pd.DataFrame(rows).set_index("Ticker")
    return df, errors


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
# Each entry: status -> (background color, text color)
STATUS_STYLE = {
    "Extreme OB": ("#ff4c4c", "#ffffff"),  # strong red, white text
    "Overbought": ("#ffb3b3", "#000000"),  # light red/pink, black text
    "Neutral":    ("#ffffff", "#000000"),  # white, black text
    "Oversold":   ("#b3ffb3", "#000000"),  # light green, black text
    "Extreme OS": ("#33cc33", "#ffffff"),  # strong green, white text
    "N/A":        ("#e0e0e0", "#666666"),
}

# Timing (momentum-based) -> (background color, text color)
TIMING_STYLE = {
    "Good":    ("#33cc33", "#ffffff"),  # green, white text
    "Neutral": ("#ffb347", "#000000"),  # orange, black text
    "Poor":    ("#ff4c4c", "#ffffff"),  # red, white text
    "N/A":     ("#e0e0e0", "#666666"),
}

# Trend is now shown as a colored arrow icon only (no background fill)
TREND_ARROWS = {
    "Bullish": "▲",
    "Bearish": "▼",
    "N/A": "–",
}

TREND_TEXT_COLORS = {
    "Bullish": "#2ecc71",  # green
    "Bearish": "#ff4c4c",  # red
    "N/A": "#999999",
}


def pct_gradient_color(val, vmin=-10, vmax=10):
    """Green for positive, yellow around zero, red/pink for negative."""
    if pd.isna(val):
        return "background-color: #e0e0e0; color: #666;"

    v = max(min(val, vmax), vmin)
    mid = 0.0

    if v >= mid:
        # 0 -> yellow, vmax -> green
        ratio = v / vmax if vmax != 0 else 0
        r = int(255 - ratio * (255 - 62))
        g = int(245 - ratio * (245 - 207))
        b = int(157 - ratio * (157 - 95))
    else:
        # vmin -> red/pink, 0 -> yellow
        ratio = v / vmin if vmin != 0 else 0  # ratio in [0,1], 1 at vmin
        r = int(255 - ratio * (255 - 255))
        g = int(245 - ratio * (245 - 77))
        b = int(157 - ratio * (157 - 77))

    text_color = "#222" if (r * 0.299 + g * 0.587 + b * 0.114) > 150 else "#fff"
    return f"background-color: rgb({r},{g},{b}); color: {text_color};"


def rsi_gradient_color(val):
    if pd.isna(val):
        bg, fg = STATUS_STYLE["N/A"]
    else:
        level = rsi_level(val)
        bg, fg = STATUS_STYLE.get(level, STATUS_STYLE["N/A"])
    return f"background-color: {bg}; color: {fg}; font-weight: 600;"


def zscore_gradient_color(val):
    """Color the Z-Score column using the same band thresholds as Status."""
    if pd.isna(val):
        bg, fg = STATUS_STYLE["N/A"]
    else:
        level = zscore_status(val)
        bg, fg = STATUS_STYLE.get(level, STATUS_STYLE["N/A"])
    return f"background-color: {bg}; color: {fg}; font-weight: 600;"


def status_color(status):
    bg, fg = STATUS_STYLE.get(status, STATUS_STYLE["N/A"])
    return f"background-color: {bg}; color: {fg}; font-weight: 600;"


def timing_color(timing):
    bg, fg = TIMING_STYLE.get(timing, TIMING_STYLE["N/A"])
    return f"background-color: {bg}; color: {fg}; font-weight: 600;"


def trend_color(trend):
    """Trend cells show only a colored arrow icon - no background fill."""
    color = TREND_TEXT_COLORS.get(trend, TREND_TEXT_COLORS["N/A"])
    return (
        f"color: {color}; background-color: transparent; "
        "font-weight: 700; font-size: 18px; text-align: center;"
    )


def styler_apply_cellwise(styler, func, **kwargs):
    """
    Apply a cell-wise style function on a pandas Styler in a way that works
    across pandas versions:
      - pandas >= 2.1.0: Styler.map (Styler.applymap was removed)
      - pandas <  2.1.0: Styler.applymap (Styler.map didn't exist yet)
    """
    if hasattr(styler, "map"):
        return styler.map(func, **kwargs)
    return styler.applymap(func, **kwargs)


def style_dataframe(df: pd.DataFrame):
    display_df = df[
        [
            "Category", "Name", "Price", "YTD %", "5-Day %", "50-DMA %", "Trend",
            "RSI (14)", "Z-Score", "Status", "Timing", "52W Low", "52W High",
        ]
    ].copy()

    styler = display_df.style
    styler = styler_apply_cellwise(styler, pct_gradient_color, subset=["YTD %"], vmin=-20, vmax=20)
    styler = styler_apply_cellwise(styler, pct_gradient_color, subset=["5-Day %"], vmin=-5, vmax=5)
    styler = styler_apply_cellwise(styler, pct_gradient_color, subset=["50-DMA %"], vmin=-10, vmax=10)
    styler = styler_apply_cellwise(styler, rsi_gradient_color, subset=["RSI (14)"])
    styler = styler_apply_cellwise(styler, zscore_gradient_color, subset=["Z-Score"])
    styler = styler_apply_cellwise(styler, status_color, subset=["Status"])
    styler = styler_apply_cellwise(styler, timing_color, subset=["Timing"])
    styler = styler_apply_cellwise(styler, trend_color, subset=["Trend"])

    styler = (
        styler
        .format(
            {
                "Price": "${:,.2f}",
                "YTD %": "{:+.2f}%",
                "5-Day %": "{:+.2f}%",
                "50-DMA %": "{:+.2f}%",
                "RSI (14)": "{:.1f}",
                "Z-Score": "{:+.2f}",
                "52W Low": "${:,.2f}",
                "52W High": "${:,.2f}",
                # Trend's underlying value stays "Bullish"/"Bearish"/"N/A"
                # for styling purposes; only the displayed text becomes an arrow.
                "Trend": lambda v: TREND_ARROWS.get(v, "–"),
            },
            na_rep="N/A",
        )
    )
    return styler


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("📈 Trend Analyzer")
st.caption(
    f"Tracking {len(INDEX_TICKERS)} major indices "
    f"({', '.join(INDEX_TICKERS)}) and {len(SECTOR_ETF_TICKERS)} sector ETFs "
    f"({', '.join(SECTOR_ETF_TICKERS)}) — "
    "price trend, momentum, and overbought/oversold status."
)

col_a, col_b = st.columns([1, 5])
with col_a:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

with st.spinner(f"Fetching market data for {len(TICKERS)} tickers..."):
    df, errors = build_dataframe(TICKERS)

if errors:
    with st.expander("⚠️ Some tickers had issues (click to view)", expanded=False):
        for e in errors:
            st.warning(e)

if df.empty or df["Price"].isna().all():
    st.error(
        "No data could be retrieved for any ticker right now. "
        "This can happen on non-trading days, during a Yahoo Finance outage, "
        "or if there's a network/connectivity issue. Please try again shortly."
    )
else:
    st.subheader("Overview")
    styled = style_dataframe(df)
    # Dynamic height so all rows are visible at once, with no internal
    # vertical scrollbar (row height ~35px + header + small padding).
    table_height = (len(df) + 1) * 35 + 3
    st.dataframe(styled, use_container_width=True, height=table_height)

    st.subheader("52-Week Trading Range")
    range_df = df[["52W Low", "Price", "52W High", "52W Position"]].copy()
    st.dataframe(
        range_df,
        use_container_width=True,
        height=table_height,
        column_config={
            "52W Low": st.column_config.NumberColumn("52W Low", format="$%.2f"),
            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "52W High": st.column_config.NumberColumn("52W High", format="$%.2f"),
            "52W Position": st.column_config.ProgressColumn(
                "Position in Range",
                help="Where the current price sits between the 52-week low and high",
                format="%.0f%%",
                min_value=0,
                max_value=1,
            ),
        },
    )

    st.caption(
        "Trend: ▲ green = price at/above the 50-day moving average, "
        "▼ red = price below it. "
        "Status (Bespoke-style 50-DMA bands): Z = (Price − SMA50) / STD50 — "
        "Extreme OB ≥ 2.0, Overbought 1.0–2.0, Neutral −1.0 to 1.0, "
        "Oversold −2.0 to −1.0, Extreme OS ≤ −2.0. "
        "Timing follows Status directly: Extreme OB/Overbought → Poor, "
        "Neutral → Neutral, Oversold/Extreme OS → Good. "
        "RSI (14) is shown separately as an additional momentum reference "
        "and is not used to compute Status or Timing."
    )

    st.caption(f"Last updated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
