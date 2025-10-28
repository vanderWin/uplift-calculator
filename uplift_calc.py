# uplift_calc.py
import math
import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import datetime as dt

# ---------- IO ----------
st.set_page_config(page_title="Organic Rank Uplift Calculator", page_icon="favicon.png", layout="wide")

st.title("Organic Rank Uplift Calculator")
st.caption(
    "This app explores the uplift in ranking positions based on keyword diffiuclty, search volume and starting rank. Traffic estimates are presented here for illustrative purposes only and should be used to help you prioritise which categories to optimise for first. For a complete opportunity analysis, establish seasonal patterns and baseline levels of brand vs. non-brand traffic."
)


DEFAULT_PATH = "test_export.csv"
DEFAULT_CTR_VALUES = [
    0.2696, 0.2030, 0.1489, 0.1114, 0.0847,
    0.0514, 0.0464, 0.0372, 0.0255, 0.0197,
    0.005, 0.004, 0.003, 0.002, 0.002,
    0.002, 0.001, 0.001, 0.001, 0.001,
]
PRIMARY_COLOR = "#AAA0FF"
DEFAULT_CATEGORY_LABEL = "Uncategorized"
DEFAULT_INTENT_LABEL = "Other Intent"
DEFAULT_RANK_CAPS = {
    "Easy": None,
    "Medium": 2,
    "Hard": 3,
    "Top10": None,
    "N/A": 2,
}

if "ctr_values" not in st.session_state:
    st.session_state["ctr_values"] = DEFAULT_CTR_VALUES.copy()
if "ctr_dialog_open" not in st.session_state:
    st.session_state["ctr_dialog_open"] = False
if "ctr_bulk_text" not in st.session_state:
    st.session_state["ctr_bulk_text"] = ""
if "ctr_bulk_error" not in st.session_state:
    st.session_state["ctr_bulk_error"] = ""

with st.sidebar:
    st.header("Data")
    upl = st.file_uploader("Upload CSV", type=["csv"])
    if upl is None:
        st.warning("Upload a CSV to get started.")
        st.stop()
    df = pd.read_csv(upl, low_memory=False, keep_default_na=False)

    # Column mapping
    st.subheader("Column mapping")
    col_keyword = st.selectbox("Keyword column", options=df.columns.tolist(), index=0)
    col_volume  = st.selectbox("Monthly search volume", options=df.columns.tolist(), index=df.columns.get_loc("VOLUME") if "VOLUME" in df.columns else 0)
    # Optional DIFFICULTY; fallback to Medium
    col_diff    = st.selectbox("Keyword Difficulty", options=["<none>"] + df.columns.tolist(),
                               index=(df.columns.get_loc("DIFFICULTY")+1) if "DIFFICULTY" in df.columns else 0)

    # Starting rank (if absent, default 100)
    guess_rank_cols = [c for c in df.columns if "RANK" in c.upper() and "[M]" in c.upper()]
    col_rank = st.selectbox("Starting rank", options=["<none>"] + df.columns.tolist(),
                            index=(df.columns.get_loc(guess_rank_cols[0])+1) if guess_rank_cols else 0)

    col_intent = st.selectbox(
        "Intent column",
        options=["<none>"] + df.columns.tolist(),
        index=(df.columns.get_loc("INTENT") + 1) if "INTENT" in df.columns else 0,
    )
    col_group = st.selectbox(
        "Main keyword group",
        options=["<none>"] + df.columns.tolist(),
        index=(df.columns.get_loc("GROUPS") + 1) if "GROUPS" in df.columns else 0,
    )

def _normalize_difficulty_label(value):
    text = str(value).strip()
    return text if text else "N/A"

if col_diff != "<none>":
    diffs_present = [_normalize_difficulty_label(v) for v in df[col_diff].unique()]
else:
    diffs_present = []

default_keys = ["Easy", "Medium", "Hard", "Top10", "N/A"]
keys = list(dict.fromkeys(default_keys + diffs_present))
default_map = {"Easy": 0.6, "Medium": 1.0, "Hard": 1.6, "Top10": 2.2, "N/A": 1.0}

# ---------- Parameters ----------
with st.sidebar:
    st.divider()
    st.subheader("Projection horizon")
    horizon = st.slider(
    "Months ahead",
    min_value=1,
    max_value=36,
    value=12,
    help="Projection window length for rank and traffic projection."
    )

    # Projection start date (default: first of next month)
    today = dt.date.today()
    if today.month == 12:
        default_proj_start = dt.date(today.year + 1, 1, 1)
    else:
        default_proj_start = dt.date(today.year, today.month + 1, 1)
    proj_start_date = st.date_input(
        "Projection Start Date",
        value=default_proj_start,
        help="Month start to anchor projections. Defaults to the first day of next month."
    )

    # Projection mode: Seasonal (default) or Average
    proj_mode = st.radio(
        "Projection Mode",
        ["Average", "Seasonal"],
        index=1,
        help="Average: use the uploaded volume for all months. Seasonal: use Google Ads 12-month seasonality per keyword."
    )

    st.divider()
    st.subheader("Rank caps by difficulty")
    st.caption("Limit how far each difficulty tier can climb. Set to 'No cap' to allow improvements to position 1.")
    cap_options = [("No cap", None)] + [(f"Position {i}", i) for i in range(1, 21)]
    rank_caps = {}
    for key in keys:
        default_cap = DEFAULT_RANK_CAPS.get(key, None)
        default_index = 0
        if default_cap is not None:
            for idx, (_, val) in enumerate(cap_options):
                if val == default_cap:
                    default_index = idx
                    break
        selection = st.selectbox(
            f"{key}",
            cap_options,
            index=default_index,
            format_func=lambda option: option[0],
            key=f"rank_cap_{key}",
        )
        rank_caps[key] = selection[1]

    st.divider()
    st.header("Phase Durations")
    st.caption("Phase durations are base months before difficulty/volume scaling.")

    # Base phase durations
    T1 = st.number_input(
    "Months to reach 50th (100 → 50)",
    min_value=0.1,
    value=1.0,
    step=0.1,
    help="Average time (months) for a keyword to move from rank 100 to 50 before scaling."
    )

    T2 = st.number_input(
        "Months to reach page 2 (50 → 20)",
        min_value=0.1,
        value=2.0,
        step=0.1,
        help="Months expected to progress from rank 50 to 20."
    )

    T3 = st.number_input(
        "Months to reach page 1 (20 → 10)",
        min_value=0.1,
        value=3.0,
        step=0.1,
        help="Months to improve from rank 20 to 10."
    )

    T4 = st.number_input(
        "Months to reach the top (10 → 1)",
        min_value=0.1,
        value=6.0,
        step=0.1,
        help="Months expected to reach position 1 once in the top 10."
    )

    k = st.number_input(
        "Curve steepness (k)",
        min_value=0.1,
        value=3.5,
        step=0.1,
        help="Controls shape of improvement within each phase. Larger k = faster early gains, slower finish."
    )

    st.divider()
    st.subheader("Difficulty multipliers")
    # Values when DIFFICULTY missing or custom categories
    st.caption("Difficulty multipliers: smaller = easier and faster ranking gains; larger = harder and slower progress.")
    m_d = {}
    for key in keys:
        m_d[key] = st.number_input(f"{key}", min_value=0.1, value=float(default_map.get(key, 1.2)), step=0.1)

    st.divider()
    st.subheader("Volume multiplier")
    st.markdown(
    "Adjusts how search volume slows growth.<br>"
    "High-volume keywords face tougher competition, so they move more slowly.<br>"
    "**Formula**: m_v = v_min + v_span × [log10(1 + volume) / log10(1 + vol_max)], clamped to [m_min, m_max].<br>"
    "_Smaller m_v → faster progress; larger m_v → slower._",
    unsafe_allow_html=True
    )


    v_min = st.number_input(
    "v_min",
    value=0.8,
    step=0.05,
    help="Baseline multiplier applied to low-volume keywords (fastest growth)."
    )

    v_span = st.number_input(
        "v_span",
        value=0.7,
        step=0.05,
        help="Range added to v_min based on keyword volume. Larger = stronger slowdown for high volume."
    )

    m_min = st.number_input(
        "Clamp m_min",
        value=0.8,
        step=0.05,
        help="Lower bound of the final volume multiplier."
    )

    m_max = st.number_input(
        "Clamp m_max",
        value=1.5,
        step=0.05,
        help="Upper bound of the final volume multiplier."
    )
    vol_max_mode = st.radio("vol_max source", ["Auto from dataset", "Manual"], index=0)
    vol_max_manual = st.number_input(
    "vol_max (manual)",
    min_value=1,
    value=100000,
    step=1000,
    format="%i",
    disabled=(vol_max_mode == "Auto from dataset"),
    help="Maximum search volume used to normalise m_v scaling."
    )

    st.divider()

# ---------- Helpers ----------

def _scaled_phase_times(vol, vol_max, diff_label, T1, T2, T3, T4):
    base = [(100,50,T1),(50,20,T2),(20,10,T3),(10,1,T4)]
    diff_str = str(diff_label).strip()
    if diff_str.upper() in {"", "N/A", "NA", "NONE"}:
        diff_key = "N/A"
    else:
        diff_key = diff_str
    md = m_d.get(diff_key, m_d.get("Medium", 1.0))
    den = math.log10(1+vol_max) if vol_max > 0 else 1.0
    norm = (math.log10(1+max(vol,0)) / den) if den > 0 else 0.0
    mv = max(m_min, min(m_max, v_min + v_span * norm))
    return [(a,b, t*md*mv) for (a,b,t) in base]

def build_phases(start_rank, vol, vol_max, diff_label):
    phases = _scaled_phase_times(vol, vol_max, diff_label, T1, T2, T3, T4)
    # pick the active phase and trim it to start_rank → b
    for (a,b,T) in phases:
        if a >= start_rank >= b:
            frac_remaining = (start_rank - b) / (a - b + 1e-9)
            T_rem = max(1e-9, T * frac_remaining)
            idx = next(i for i,p in enumerate(phases) if p==(a,b,T))
            return [(start_rank, b, T_rem)] + phases[idx+1:]
    # If already <10, do the tail of 10→1 only
    if start_rank < 10:
        a, b, T = 10, 1, next(t for (aa,bb,t) in phases if aa==10 and bb==1)
        frac_remaining = (start_rank - b) / (a - b + 1e-9)
        return [(start_rank, b, max(1e-9, T * frac_remaining))]
    return phases


def rank_at_month(t, start_rank, vol, vol_max, diff_label, k=3.5):
    phases = build_phases(start_rank, vol, vol_max, diff_label)
    rem = t
    cur = float(start_rank)
    for (a,b,T) in phases:
        if rem <= 0:
            return cur
        if rem >= T:
            cur = float(b)
            rem -= T
        else:
            frac = rem / max(T,1e-9)
            r = b + (a - b) * math.exp(-k * frac)  # always moves toward b
            return min(cur, r)  # monotonic improvement
    return cur


def normalize_ctr_values(values, n=20):
    arr = [max(0.0, float(x)) for x in values[:n]]
    if len(arr) < n:
        arr += [0.0] * (n - len(arr))
    return arr


def parse_ctr_bulk_input(blob, expected=20):
    """Parse pasted CTR percentages and return a list of floats in percent units."""
    if not blob:
        return [], "No values provided."

    # Replace common separators with whitespace to make splitting reliable
    cleaned_text = blob.replace(",", " ").strip()
    # Extract numeric components (allows for values like 6.0% or 0.62)
    tokens = re.findall(r"-?\d+(?:\.\d+)?", cleaned_text.replace("%", " "))
    values = []
    for token in tokens:
        try:
            values.append(round(float(token), 2))
        except ValueError:
            return [], f"Unable to read value '{token}'."

    if len(values) != expected:
        return [], f"Expected {expected} values, but found {len(values)}."

    if any(v < 0 for v in values):
        return [], "CTR values must be zero or greater."

    return values, ""


def ctr_bulk_dialog(expected=20):
    st.markdown("#### Bulk paste CTR values")
    st.write("Paste CTR percentages (one per line). Percent signs are optional.")

    st.text_area(
        "CTR values",
        key="ctr_bulk_text",
        height=220,
        placeholder="6.00%\n3.19%\n...",
    )

    error_msg = st.session_state.get("ctr_bulk_error", "")
    if error_msg:
        st.error(error_msg)

    apply_col, close_col = st.columns(2)
    apply_clicked = apply_col.button(
        "Apply",
        use_container_width=True,
        key="ctr_bulk_apply",
    )
    close_clicked = close_col.button(
        "Close",
        use_container_width=True,
        key="ctr_bulk_close",
    )

    if apply_clicked:
        values, error = parse_ctr_bulk_input(
            st.session_state.get("ctr_bulk_text", ""),
            expected=expected,
        )
        if error:
            st.session_state["ctr_bulk_error"] = error
            st.session_state["ctr_dialog_open"] = True
            return

        decimals = [round(v / 100.0, 4) for v in values]
        st.session_state["ctr_values"] = normalize_ctr_values(decimals, expected)
        st.session_state["ctr_bulk_error"] = ""
        st.session_state["ctr_dialog_open"] = False

    if close_clicked:
        st.session_state["ctr_dialog_open"] = False
        st.session_state["ctr_bulk_error"] = ""

def ctr_top20(rank, arr):
    r = int(round(rank))
    if 1 <= r <= len(arr):
        return float(arr[r-1])
    return 0.0


# ---------- Prepare data ----------
selected_cols = [col_keyword, col_volume]
if col_diff != "<none>":
    selected_cols.append(col_diff)
if col_rank != "<none>":
    selected_cols.append(col_rank)
if col_intent != "<none>":
    selected_cols.append(col_intent)
if col_group != "<none>":
    selected_cols.append(col_group)

selected_cols = list(dict.fromkeys(selected_cols))
work = df[selected_cols].copy()

rename_map = {col_keyword: "KEYWORD", col_volume: "VOLUME"}
if col_diff != "<none>":
    rename_map[col_diff] = "DIFFICULTY"
if col_rank != "<none>":
    rename_map[col_rank] = "START_RANK"
if col_intent != "<none>":
    rename_map[col_intent] = "INTENT"
if col_group != "<none>":
    rename_map[col_group] = "CATEGORY"

work.rename(columns=rename_map, inplace=True)

if "DIFFICULTY" not in work:
    work["DIFFICULTY"] = "N/A"
else:
    work["DIFFICULTY"] = work["DIFFICULTY"].fillna("N/A")
if "START_RANK" not in work:
    work["START_RANK"] = 100
if "INTENT" not in work:
    work["INTENT"] = DEFAULT_INTENT_LABEL
if "CATEGORY" not in work:
    work["CATEGORY"] = DEFAULT_CATEGORY_LABEL

work["INTENT"] = work["INTENT"].fillna(DEFAULT_INTENT_LABEL).astype(str)
work["CATEGORY"] = work["CATEGORY"].fillna(DEFAULT_CATEGORY_LABEL).astype(str)
work["MAIN_CATEGORY"] = (
    work["CATEGORY"]
    .str.replace(r"\s*\|.*", "", regex=True)
    .str.strip()
)
work["MAIN_CATEGORY"] = work["MAIN_CATEGORY"].replace("", DEFAULT_CATEGORY_LABEL).fillna(DEFAULT_CATEGORY_LABEL)
work["DIFFICULTY"] = (
    work["DIFFICULTY"]
    .astype(str)
    .str.strip()
    .replace({"": "N/A", "nan": "N/A", "None": "N/A", "NONE": "N/A"})
)
work.loc[work["DIFFICULTY"].str.upper().isin({"NA", "N/A"}), "DIFFICULTY"] = "N/A"

# clean types
work["VOLUME"] = pd.to_numeric(work["VOLUME"], errors="coerce").fillna(0).clip(lower=0)
work["START_RANK"] = pd.to_numeric(work["START_RANK"], errors="coerce").fillna(100).clip(lower=1, upper=200)

vol_max = work["VOLUME"].max() if vol_max_mode == "Auto from dataset" else float(vol_max_manual)
model_signature = (
    float(T1),
    float(T2),
    float(T3),
    float(T4),
    float(k),
    float(vol_max),
    vol_max_mode,
    float(v_min),
    float(v_span),
    float(m_min),
    float(m_max),
    col_intent,
    col_group,
    tuple(sorted((str(key), float(val)) for key, val in m_d.items())),
    tuple(sorted((str(key), None if val is None else float(val)) for key, val in rank_caps.items())),
)

# ===== Seasonal search volumes via Google Ads (optional) =====
# We lazily fetch on demand, batch requests, and cache results in session state.

def _first_of_month(d: dt.date) -> dt.date:
    return dt.date(d.year, d.month, 1)

def _add_months(d: dt.date, n: int) -> dt.date:
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    return dt.date(y, m, 1)

def _month_starts(start: dt.date, horizon: int) -> list[dt.date]:
    return [_add_months(_first_of_month(start), i) for i in range(horizon + 1)]

# Default targeting if none provided via secrets
_DEFAULT_GEO_IDS = ["2826"]  # UK
_DEFAULT_LANGUAGE_ID = "1000"  # English

def get_gads_client_and_customer_id():
    """Prefer Streamlit Secrets; safely fall back to local google-ads.yaml.
    Returns (client, effective_customer_id) or (None, "").
    """
    try:
        from google.ads.googleads.client import GoogleAdsClient
    except Exception:
        st.error("google-ads library not installed. Run: pip install google-ads")
        return None, ""

    def _norm_id(s):
        return str(s).replace("-", "").strip() if s else ""

    try:
        has_secrets = "google_ads" in st.secrets
    except Exception:
        has_secrets = False

    if has_secrets:
        s = st.secrets["google_ads"]
        try:
            cfg = {
                "developer_token": s.get("developer_token"),
                "client_id": s.get("client_id"),
                "client_secret": s.get("client_secret"),
                "refresh_token": s.get("refresh_token"),
                "login_customer_id": _norm_id(s.get("login_customer_id")),
                "client_customer_id": _norm_id(s.get("client_customer_id")),
                "use_proto_plus": True,
            }
            cfg = {k: v for k, v in cfg.items() if v is not None}
            import yaml as _yaml
            yaml_text = _yaml.dump(cfg)
            client = GoogleAdsClient.load_from_string(yaml_text, version="v20")
            effective_id = _norm_id(s.get("client_customer_id")) or _norm_id(s.get("login_customer_id"))
            return client, effective_id
        except Exception as e:
            st.error(f"Failed to load Google Ads client from Streamlit secrets: {e}")
            return None, ""

    # Fallback: local yaml
    from pathlib import Path as _Path
    yaml_path = _Path(__file__).parent / "google-ads.yaml"
    if not yaml_path.exists():
        st.error("No Streamlit secrets and no local google-ads.yaml found.")
        return None, ""
    try:
        import yaml as _yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = _yaml.safe_load(f) or {}
        client = GoogleAdsClient.load_from_storage(str(yaml_path), version="v20")
        effective_id = _norm_id(cfg.get("client_customer_id")) or _norm_id(cfg.get("login_customer_id"))
        return client, effective_id
    except Exception as e:
        st.error(f"Failed to load Google Ads client from {yaml_path.name}: {e}")
        return None, ""

def _parse_geo_ids(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def fetch_historical_metrics_gads(client, customer_id: str, keywords: list[str],
                                  geo_ids: list[str], language_id: str,
                                  batch_size: int = 700) -> tuple[pd.DataFrame, list]:
    try:
        from google.protobuf.json_format import MessageToDict
    except Exception:
        MessageToDict = None

    googleads_service = client.get_service("GoogleAdsService")
    idea_service = client.get_service("KeywordPlanIdeaService")

    out_rows, raw_results = [], []
    total = len(keywords)
    prog = st.progress(0.0, text="Requesting batches…")

    for i in range(0, total, batch_size):
        batch = keywords[i:i+batch_size]
        req = client.get_type("GenerateKeywordHistoricalMetricsRequest")
        req.customer_id = customer_id
        req.keywords.extend(batch)
        req.keyword_plan_network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH
        req.language = googleads_service.language_constant_path(language_id)
        for gid in geo_ids:
            req.geo_target_constants.append(googleads_service.geo_target_constant_path(gid))

        resp = idea_service.generate_keyword_historical_metrics(request=req)

        if MessageToDict:
            raw_results.extend([MessageToDict(r._pb) for r in resp.results])
        else:
            for r in resp.results:
                raw_results.append({"text": getattr(r, "text", ""), "closeVariants": list(getattr(r, "close_variants", []))})

        for r in resp.results:
            m = r.keyword_metrics
            canonical = r.text.lower().strip()
            variants = [v.lower().strip() for v in (list(r.close_variants) if r.close_variants else [])]
            aliases = [canonical] + [v for v in variants if v]
            base = {
                "canonical_keyword": canonical,
                "aliases": aliases,
                "close_variants": ", ".join(variants) if variants else "",
                "avg_monthly_searches": int(m.avg_monthly_searches) if m.avg_monthly_searches is not None else None,
                "competition_index": int(m.competition_index) if m.competition_index is not None else None,
                "competition_level": m.competition.name if hasattr(m.competition, "name") else str(m.competition),
                "low_top_of_page_bid_micros": int(m.low_top_of_page_bid_micros) if m.low_top_of_page_bid_micros is not None else None,
                "high_top_of_page_bid_micros": int(m.high_top_of_page_bid_micros) if m.high_top_of_page_bid_micros is not None else None,
            }
            if getattr(m, "monthly_search_volumes", None):
                for mv in m.monthly_search_volumes:
                    month_num = [
                        "JANUARY","FEBRUARY","MARCH","APRIL","MAY","JUNE",
                        "JULY","AUGUST","SEPTEMBER","OCTOBER","NOVEMBER","DECEMBER"
                    ].index(mv.month.name)+1
                    out_rows.append(base | {
                        "year": int(mv.year),
                        "month": month_num,
                        "monthly_searches": int(mv.monthly_searches) if mv.monthly_searches is not None else None
                    })
            else:
                out_rows.append(base | {"year": None, "month": None, "monthly_searches": None})

        prog.progress(min((i+batch_size)/max(total,1),1.0))

    df = pd.DataFrame(out_rows)
    if not df.empty and "aliases" in df.columns:
        df = df.explode("aliases", ignore_index=True).rename(columns={"aliases": "keyword_norm"})
    else:
        df["keyword_norm"] = ""
    return df, raw_results

# When in Seasonal mode, ensure results are available and cached
seasonality_key = None
if proj_mode == "Seasonal":
    # Build the keyword list to query
    to_query = (
        work["KEYWORD"].astype(str).str.strip().str.lower()
        .replace("", pd.NA).dropna().unique().tolist()
    )

    # Derive default geo/language from secrets or defaults
    geo_ids = _DEFAULT_GEO_IDS
    lang_id = _DEFAULT_LANGUAGE_ID
    try:
        if "google_ads" in st.secrets:
            gs = st.secrets["google_ads"]
            if gs.get("geo_target_ids"):
                geo_ids = _parse_geo_ids(str(gs.get("geo_target_ids")))
            if gs.get("language_id"):
                lang_id = str(gs.get("language_id"))
    except Exception:
        pass

    seasonality_key = ("gads_v20_seasonal", tuple(sorted(to_query)), tuple(sorted(geo_ids)), lang_id)
    if st.session_state.get("gads_results_key") != seasonality_key:
        client, effective_id = get_gads_client_and_customer_id()
        if client and effective_id:
            with st.spinner("Loading Search Volumes from Google Ads…"):
                gads_df, raw_json = fetch_historical_metrics_gads(client, effective_id, to_query, geo_ids, lang_id)
            st.session_state.gads_results_key = seasonality_key
            st.session_state.gads_results_df = gads_df
            st.session_state.gads_raw_json = raw_json
        else:
            st.warning("Google Ads credentials not available. Using average volumes instead.")
            proj_mode = "Average"

# Extend model signature to account for projection mode and seasonality inputs
model_signature = model_signature + (
    str(proj_mode),
    str(proj_start_date.isoformat()),
    seasonality_key,
)

# selection
r1_col1, r1_col2 = st.columns(2)
with r1_col1:
    st.subheader("Sample keywords")
    sample_df = work[["KEYWORD", "VOLUME", "DIFFICULTY", "START_RANK"]]
    sample_df = (
        sample_df.sort_values("VOLUME", ascending=False)
        .head(5000)
        .reset_index(drop=True)
        .rename(
            columns={
                "KEYWORD": "Keyword",
                "VOLUME": "Volume",
                "DIFFICULTY": "Difficulty",
                "START_RANK": "Start Rank",
            }
        )
    )

    if sample_df.empty:
        st.info("No keywords available. Adjust your filters or upload a different file.")
        st.stop()

    if "selected_keyword" not in st.session_state:
        st.session_state["selected_keyword"] = sample_df.iloc[0]["Keyword"]

    current_keyword = st.session_state.get("selected_keyword")
    if current_keyword not in work["KEYWORD"].values:
        current_keyword = sample_df.iloc[0]["Keyword"]
        st.session_state["selected_keyword"] = current_keyword
    current_row = work.loc[work["KEYWORD"] == current_keyword].iloc[0].to_dict()

    st.markdown(
        f"**Selected keyword:** {current_row['KEYWORD']}<br>"
        f"Volume: {int(current_row['VOLUME']):,} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Difficulty: {current_row['DIFFICULTY']} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Start Rank: {int(current_row['START_RANK'])}",
        unsafe_allow_html=True,
    )

    table_event = st.dataframe(
        sample_df,
        use_container_width=True,
        hide_index=True,
        height=320,
        selection_mode="single-row",
        on_select="rerun",
        key="sample_keyword_table",
        column_config={
            "Keyword": st.column_config.TextColumn("Keyword"),
            "Volume": st.column_config.NumberColumn("Volume", format="%d"),
            "Difficulty": st.column_config.TextColumn("Difficulty"),
            "Start Rank": st.column_config.NumberColumn("Start Rank", format="%d"),
        },
    )

    selection = getattr(table_event, "selection", None)
    selected_rows = selection.get("rows", []) if isinstance(selection, dict) else []
    if selected_rows:
        sel_idx = selected_rows[0]
        if 0 <= sel_idx < len(sample_df):
            st.session_state["selected_keyword"] = sample_df.iloc[sel_idx]["Keyword"]

    current_keyword = st.session_state.get("selected_keyword", current_keyword)
    if current_keyword not in work["KEYWORD"].values:
        current_keyword = sample_df.iloc[0]["Keyword"]
        st.session_state["selected_keyword"] = current_keyword
    row = work.loc[work["KEYWORD"] == current_keyword].iloc[0].to_dict()

with r1_col2:
    st.subheader("Click through curve (CTR)")
    st.caption("Adjust CTR for ranks 1-20; changes update all projections.")

    if st.button("Bulk paste CTR values", key="open_ctr_dialog"):
        st.session_state["ctr_dialog_open"] = True
        st.session_state["ctr_bulk_error"] = ""
        st.session_state["ctr_bulk_text"] = "\n".join(
            f"{v * 100:.2f}%" for v in st.session_state["ctr_values"]
        )

    if st.session_state.get("ctr_dialog_open"):
        ctr_bulk_dialog(expected=20)

    ctr_table_df = pd.DataFrame({
        "Rank": list(range(1, 21)),
        "CTR (%)": [v * 100 for v in st.session_state["ctr_values"]],
    })
    edited_ctr_df = st.data_editor(
        ctr_table_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="ctr_editor",
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", disabled=True),
            "CTR (%)": st.column_config.NumberColumn(
                "CTR (%)", format="%.2f", step=0.01, min_value=0.0, max_value=100.0
            ),
        },
    )

    updated_ctr_vals = []
    for val in edited_ctr_df["CTR (%)"]:
        try:
            updated_ctr_vals.append(float(val) / 100.0)
        except (TypeError, ValueError):
            updated_ctr_vals.append(0.0)
    st.session_state["ctr_values"] = normalize_ctr_values(updated_ctr_vals, 20)

ctr_values = st.session_state["ctr_values"]

# ---------- Single keyword projection ----------
def project_keyword(row, months):
    diff_label = str(row["DIFFICULTY"])
    cap_value = rank_caps.get(diff_label, None)
    if cap_value is not None:
        try:
            cap_value = max(1.0, float(cap_value))
        except (TypeError, ValueError):
            cap_value = None

    start_rank_value = float(row["START_RANK"])
    if cap_value is not None and start_rank_value <= cap_value:
        cap_value = None

    ranks = []
    for t in range(months+1):
        r = rank_at_month(
            t,
            start_rank=int(row["START_RANK"]),
            vol=row["VOLUME"],
            vol_max=vol_max,
            diff_label=row["DIFFICULTY"],
            k=k
        )
        r = max(1.0, float(r))
        if cap_value is not None:
            r = max(r, cap_value)
        ranks.append(r)
    return np.array(ranks)



months = list(range(horizon+1))
# Month starts as actual calendar dates for display
month_starts_preview = _month_starts(proj_start_date, horizon)
row_proj = project_keyword(row, horizon)

r2_col1, r2_col2 = st.columns(2)

with r2_col1:
    st.subheader("Rank trajectory (selected keyword)")
    chart_df = pd.DataFrame({"Month": month_starts_preview, "Rank": row_proj})
    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("yearmonth(Month):T", title=None),
            y=alt.Y("Rank:Q", scale=alt.Scale(reverse=True)),
            tooltip=[
                alt.Tooltip("yearmonth(Month):T", title="Month"),
                alt.Tooltip("Rank:Q", title="Rank"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)

with r2_col2:
    st.subheader("CTR curve")
    ctr_chart_df = pd.DataFrame({"Rank": list(range(1, 21)), "CTR": ctr_values})
    ctr_chart = (
        alt.Chart(ctr_chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Rank:Q", axis=alt.Axis(title="Rank", tickMinStep=1)),
            y=alt.Y("CTR:Q", axis=alt.Axis(title="CTR", format="%")),
            tooltip=[
                alt.Tooltip("Rank:Q", title="Rank"),
                alt.Tooltip("CTR:Q", title="CTR", format=".2%"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(ctr_chart, use_container_width=True)
    

# Always-on CTR -> visits for the selected keyword
CTR_ARR = normalize_ctr_values(st.session_state["ctr_values"], 20)
row_ctr = np.array([ctr_top20(r, CTR_ARR) for r in row_proj])
row_visits = (row_ctr * row["VOLUME"]).round(2)

st.subheader("Expected visits (selected keyword)")
visits_df = pd.DataFrame({"Month": month_starts_preview, "Visits": row_visits})
visits_chart = (
    alt.Chart(visits_df)
    .mark_bar()
    .encode(
        x=alt.X("yearmonth(Month):O", title=None),
        y=alt.Y("Visits:Q", title="Visits"),
        tooltip=[
            alt.Tooltip("yearmonth(Month):T", title="Month"),
            alt.Tooltip("Visits:Q", title="Visits", format=",.0f"),
        ],
    )
    .properties(height=220)
)
st.altair_chart(visits_chart, use_container_width=True)



# ---------- Batch projection + export ----------
st.divider()

@st.cache_data(show_spinner=False)
def batch_forecast(work_df, horizon, ctr_arr_tuple, model_signature):
    _ = model_signature  # ensures cache invalidates when model settings change
    ctr_arr = tuple(ctr_arr_tuple)
    out_rows = []

    # Precompute month start dates aligned to projection start
    month_starts = [dt.date(proj_start_date.year, proj_start_date.month, 1)]
    for i in range(1, horizon + 1):
        y = month_starts[-1].year + (month_starts[-1].month // 12)
        m = (month_starts[-1].month % 12) + 1
        month_starts.append(dt.date(y, m, 1))

    # Seasonal volumes if available
    use_seasonal = (str(proj_mode).lower() == "seasonal") and (st.session_state.get("gads_results_df") is not None)
    gdf = st.session_state.get("gads_results_df") if use_seasonal else None
    seasonal_cache: dict[str, np.ndarray] = {}

    def seasonal_vector_for_keyword(keyword: str, fallback_avg: float) -> np.ndarray:
        key = str(keyword).lower().strip()
        if key in seasonal_cache:
            return seasonal_cache[key]
        vec = np.array([fallback_avg] * (horizon + 1), dtype=float)
        if gdf is None or gdf.empty:
            seasonal_cache[key] = vec
            return vec
        sub = gdf[gdf["keyword_norm"] == key]
        if sub.empty:
            seasonal_cache[key] = vec
            return vec
        try:
            month_map = (
                sub.dropna(subset=["month", "monthly_searches"])
                   .groupby("month", as_index=False)["monthly_searches"].sum()
            )
            month_to_value = {int(r["month"]): float(r["monthly_searches"]) for _, r in month_map.iterrows()}
            vals = []
            for d in month_starts:
                mv = month_to_value.get(d.month)
                if mv is None or mv <= 0:
                    mv = float(fallback_avg)
                vals.append(mv)
            vec = np.array(vals, dtype=float)
        except Exception:
            vec = np.array([fallback_avg] * (horizon + 1), dtype=float)
        seasonal_cache[key] = vec
        return vec

    for _, rw in work_df.iterrows():
        proj = project_keyword(rw, horizon)
        ctr_vals = np.array([ctr_top20(r, ctr_arr) for r in proj])

        # Per-month volume: seasonal or constant average
        if use_seasonal:
            vols = seasonal_vector_for_keyword(rw["KEYWORD"], float(rw["VOLUME"]))
        else:
            vols = np.array([float(rw["VOLUME"]) for _ in range(horizon + 1)], dtype=float)

        visits = (ctr_vals * vols).round(4)
        baseline_ctr = ctr_top20(rw["START_RANK"], ctr_arr)
        baseline_visits_vec = (baseline_ctr * vols).astype(float)
        for m, rnk, vs, c, base_vs in zip(months, proj, visits, ctr_vals, baseline_visits_vec):
            uplift_vs = float(vs) - float(base_vs)
            out_rows.append({
                "KEYWORD": rw["KEYWORD"],
                "VOLUME": int(rw["VOLUME"]),
                "DIFFICULTY": rw["DIFFICULTY"],
                "START_RANK": int(rw["START_RANK"]),
                "CATEGORY": rw.get("CATEGORY", DEFAULT_CATEGORY_LABEL),
                "MAIN_CATEGORY": rw.get("MAIN_CATEGORY", DEFAULT_CATEGORY_LABEL),
                "INTENT": rw.get("INTENT", DEFAULT_INTENT_LABEL),
                "MONTH_AHEAD": m,
                "MONTH_START": month_starts[m],
                "PRED_RANK": round(float(rnk), 2),
                "EXP_CTR": round(float(c), 5),
                "EXP_VISITS": round(float(vs), 2),
                "BASELINE_VISITS": round(float(base_vs), 2),
                "EXP_UPLIFT": round(float(uplift_vs), 2),
            })
    out = pd.DataFrame(out_rows)
    future_rows = out[out["MONTH_AHEAD"] > 0].copy()

    agg_total = (future_rows
                 .groupby("MONTH_START", as_index=False)
                 .agg(
                     TOTAL_UPLIFT=("EXP_UPLIFT", "sum"),
                     TOTAL_VISITS=("EXP_VISITS", "sum"),
                     TOTAL_BASELINE=("BASELINE_VISITS", "sum"),
                 ))

    cat_df = (
        future_rows
        .groupby(["MONTH_START", "MAIN_CATEGORY"], as_index=False)
        .agg(UPLIFT=("EXP_UPLIFT", "sum"))
        .rename(columns={"MAIN_CATEGORY": "CATEGORY"})
    )
    if not cat_df.empty:
        cat_df = cat_df.sort_values(["MONTH_START", "CATEGORY"])
    cat_df["CATEGORY"] = cat_df["CATEGORY"].fillna(DEFAULT_CATEGORY_LABEL).astype(str)
    if cat_df.empty:
        cat_pivot = pd.DataFrame(columns=["MONTH_START"])
    else:
        cat_pivot = (
            cat_df.pivot(index="MONTH_START", columns="CATEGORY", values="UPLIFT")
            .fillna(0.0)
            .reset_index()
        )

    intent_df = (
        future_rows
        .groupby(["MONTH_START", "INTENT"], as_index=False)
        .agg(UPLIFT=("EXP_UPLIFT", "sum"))
    )
    if intent_df.empty:
        intent_pivot = pd.DataFrame(columns=["MONTH_START"])
    else:
        intent_df = intent_df.sort_values(["MONTH_START", "INTENT"])
        intent_df["INTENT"] = intent_df["INTENT"].fillna(DEFAULT_INTENT_LABEL).astype(str)
        intent_pivot = (
            intent_df.pivot(index="MONTH_START", columns="INTENT", values="UPLIFT")
            .fillna(0.0)
            .reset_index()
        )

    return out, agg_total, cat_pivot, cat_df, intent_pivot, intent_df


with st.spinner("Running projections..."):
    proj_df, agg_month_df, cat_pivot, cat_long, intent_pivot, intent_long = batch_forecast(
        work, horizon, tuple(CTR_ARR), model_signature
    )

total_uplift_sum = float(agg_month_df["TOTAL_UPLIFT"].sum()) if not agg_month_df.empty else 0.0
total_baseline_sum = float(agg_month_df["TOTAL_BASELINE"].sum()) if not agg_month_df.empty else 0.0
additional_label = (
    f"{total_uplift_sum/1000:.1f}K additional traffic over {horizon} months"
    if total_uplift_sum else f"0 additional traffic over {horizon} months"
)
uplift_pct = (total_uplift_sum / total_baseline_sum * 100) if total_baseline_sum > 0 else 0.0
uplift_pct_label = f"{uplift_pct:.1f}% uplift in traffic over {horizon} months"

header_col_left, header_col_right = st.columns([1, 1])
with header_col_left:
    st.header("Batch projection")
with header_col_right:
    st.markdown(
        f"<div style='text-align:right; font-size:1rem;'><strong>{additional_label}</strong><br>{uplift_pct_label}</div>",
        unsafe_allow_html=True,
    )
st.caption("Tip: tune phase durations, difficulty multipliers, and CTR to see realism vs. speed. Lower k fattens the curve; higher k front-loads gains.")

st.subheader("Uplift Table")

category_order = []
if not cat_long.empty:
    category_order = (
        cat_long.groupby("CATEGORY")["UPLIFT"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
cat_columns = [col for col in cat_pivot.columns if col != "MONTH_START"]
if category_order:
    ordered_cols = [c for c in category_order if c in cat_columns]
    if ordered_cols:
        cat_pivot = cat_pivot[["MONTH_START"] + ordered_cols]
        cat_columns = ordered_cols

table_df = agg_month_df.merge(cat_pivot, on="MONTH_START", how="left")
table_df = table_df.fillna(0.0).sort_values("MONTH_START")

display_df = table_df.rename(
    columns={
        "MONTH_START": "Month",
        "TOTAL_UPLIFT": "Total uplift",
        "TOTAL_VISITS": "Total visits",
        "TOTAL_BASELINE": "Baseline visits",
    }
)
try:
    # Ensure proper date dtype for Streamlit so it doesn't render epoch numbers
    display_df["Month"] = pd.to_datetime(display_df["Month"])
except Exception:
    # Fallback: leave as-is if conversion fails
    pass
category_display_cols = []
for col in cat_columns:
    display_name = f"{col} uplift"
    display_df = display_df.rename(columns={col: display_name})
    category_display_cols.append(display_name)

table_column_config = {
    "Month": st.column_config.DateColumn("Month", format="MMM YYYY"),
    "Total uplift": st.column_config.NumberColumn("Total uplift", format="%.0f"),
    "Total visits": st.column_config.NumberColumn("Total visits", format="%.0f"),
    "Baseline visits": st.column_config.NumberColumn("Baseline visits", format="%.0f"),
}
for col in category_display_cols:
    table_column_config[col] = st.column_config.NumberColumn(col, format="%.0f")

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    height=320,
    column_config=table_column_config,
)

st.write("")
st.subheader("Total Uplift by Month")
st.caption("Baseline uses average volumes in Average mode, or monthly Google Ads seasonality in Seasonal mode.")
if not agg_month_df.empty:
    chart_data = agg_month_df.sort_values("MONTH_START")
    def _format_delta(v):
        prefix = "+" if v >= 0 else ""
        if abs(v) >= 1000:
            return f"{prefix}{v/1000:.1f}K"
        return f"{prefix}{v:.0f}"

    chart_data["TOTAL_UPLIFT_LABEL"] = chart_data["TOTAL_UPLIFT"].apply(_format_delta)
    chart_data["TOTAL_COMBINED"] = chart_data["TOTAL_BASELINE"] + chart_data["TOTAL_UPLIFT"]

    chart_long = chart_data.melt(
        id_vars=["MONTH_START"],
        value_vars=["TOTAL_BASELINE", "TOTAL_UPLIFT"],
        var_name="Series",
        value_name="Visits",
    )
    chart_long = chart_long.merge(
        chart_data[["MONTH_START", "TOTAL_UPLIFT", "TOTAL_BASELINE", "TOTAL_VISITS"]],
        on="MONTH_START",
        how="left",
    )
    chart_long["Series_label"] = chart_long["Series"].map(
        {"TOTAL_BASELINE": "Baseline", "TOTAL_UPLIFT": "Uplift"}
    )
    chart_long["Series_order"] = chart_long["Series_label"].map({"Baseline": 0, "Uplift": 1})

    color_scale = alt.Scale(domain=["Baseline", "Uplift"], range=["#4A4A4A", PRIMARY_COLOR])

    stack_chart = (
        alt.Chart(chart_long)
        .mark_bar()
        .encode(
            x=alt.X("yearmonth(MONTH_START):O", title=None),
            y=alt.Y("Visits:Q", stack="zero", title="Visits"),
            color=alt.Color("Series_label:N", scale=color_scale, title="Traffic component"),
            order=alt.Order("Series_order:Q"),
            tooltip=[
                alt.Tooltip("yearmonth(MONTH_START):T", title="Month"),
                alt.Tooltip("Series_label:N", title="Component"),
                alt.Tooltip("Visits:Q", title="Visits", format=",.0f"),
                alt.Tooltip("TOTAL_BASELINE:Q", title="Baseline visits", format=",.0f"),
                alt.Tooltip("TOTAL_UPLIFT:Q", title="Uplift", format=",.0f"),
                alt.Tooltip("TOTAL_VISITS:Q", title="Projected visits", format=",.0f"),
            ],
        )
    )

    base_chart = alt.Chart(chart_data)
    label_chart = (
        base_chart
        .mark_text(align="center", baseline="bottom", dy=-4, color="#2B0573", fontSize=12)
        .encode(
            x=alt.X("yearmonth(MONTH_START):O", title=None),
            y=alt.Y("TOTAL_COMBINED:Q"),
            text="TOTAL_UPLIFT_LABEL:N",
        )
    )

    combined_chart = (stack_chart + label_chart).properties(height=260)
    st.altair_chart(combined_chart, use_container_width=True)
else:
    st.info("No uplift data available for the current configuration.")

st.write("")
st.subheader("Uplift by Category")
if not cat_long.empty:
    cat_chart = (
        alt.Chart(cat_long)
        .mark_bar()
        .encode(
            x=alt.X("yearmonth(MONTH_START):O", title=None),
            y=alt.Y("UPLIFT:Q", title="Uplift"),
            color=alt.Color("CATEGORY:N", title="Category"),
            tooltip=[
                alt.Tooltip("yearmonth(MONTH_START):T", title="Month"),
                alt.Tooltip("CATEGORY:N", title="Category"),
                alt.Tooltip("UPLIFT:Q", title="Uplift", format=",.0f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(cat_chart, use_container_width=True)
else:
    st.info("No category data available for projection.")

st.write("")
st.subheader("Uplift by Intent")
if not intent_long.empty:
    intent_chart = (
        alt.Chart(intent_long)
        .mark_bar()
        .encode(
            x=alt.X("yearmonth(MONTH_START):O", title=None),
            y=alt.Y("UPLIFT:Q", title="Uplift"),
            color=alt.Color("INTENT:N", title="Intent"),
            tooltip=[
                alt.Tooltip("yearmonth(MONTH_START):T", title="Month"),
                alt.Tooltip("INTENT:N", title="Intent"),
                alt.Tooltip("UPLIFT:Q", title="Uplift", format=",.0f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(intent_chart, use_container_width=True)
else:
    st.info("No intent data available for projection.")

st.download_button(
    "Download detailed projections (CSV)",
    data=proj_df.to_csv(index=False).encode("utf-8"),
    file_name="rank_traffic_projection.csv",
    mime="text/csv"
)
# ===== Centered footer (replaces sidebar logo & copyright) =====
import base64, mimetypes, datetime, os
from pathlib import Path

def _data_uri_for_logo():
    # try common locations / names
    candidates = [
        Path("logo.svg"), Path("logo.png"),
        Path("static/logo.svg"), Path("static/logo.png")
    ]
    for p in candidates:
        if p.exists():
            mime = mimetypes.guess_type(p.name)[0] or "image/png"
            b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
            return f"data:{mime};base64,{b64}"
    return None  # fallback: no logo found

def render_footer(bg="#2b0573", max_h=70):
    current_year = datetime.datetime.now().year
    logo_uri = _data_uri_for_logo()

    img_html = (
        f'<img src="{logo_uri}" style="max-height:{max_h}px; width:auto; margin-bottom:0px;" />'
        if logo_uri else ""
    )

    st.markdown(
        f"""
        <div style="background:{bg}; padding:5px; text-align:center; margin-top:40px; border-radius:10px; ">
            {img_html}
            <div style="color:#fff; font-size:0.9em;">
                &copy; {current_year}
                <a href="https://www.journeyfurther.com/?utm_source=ranking_uplift_calculator&utm_medium=footer&utm_campaign=ranking_uplift_calculator"
                   target="_blank" style="color:#fff; text-decoration:none;">Journey Further</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

render_footer()  # call at the very end of the script
