# pvc_app/pvc.py

import json
import math
from io import BytesIO
from datetime import date

import pandas as pd
from dateutil.relativedelta import relativedelta
from flask import (
    Blueprint,
    request,
    redirect,
    url_for,
    flash,
    render_template,
    send_file,
    jsonify,
)
from flask_login import login_required, current_user

from . import db
from .models import Item, ItemIndex, PVCResult, TenderMaster, TenderVendor

GST_FACTOR = 1.18

pvc_bp = Blueprint("pvc", __name__)


# ───────────────────────────────────
# Generic helpers
# ───────────────────────────────────

def safe_float(x):
    try:
        return float(str(x or 0).replace(",", "").strip())
    except Exception:
        return 0.0


def safe_round(x, n=2):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, n)
    except Exception:
        return None


def nan_to_none(value):
    """
    Convert float NaN / Inf to None (maps to SQL NULL).
    Also replaces NaN / Infinity literals inside JSON strings with null.
    """
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, str):
        return (
            value
            .replace(": NaN",       ": null")
            .replace(":NaN",        ":null")
            .replace(": Infinity",  ": null")
            .replace(":Infinity",   ":null")
            .replace(": -Infinity", ": null")
            .replace(":-Infinity",  ":null")
        )
    return value


def to_month_start(d):
    if not d:
        return pd.NaT
    ts = pd.to_datetime(d, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return pd.Timestamp(ts.year, ts.month, 1)


def previous_month(d, n=1):
    d = to_month_start(d)
    return pd.NaT if pd.isna(d) else d - relativedelta(months=n)


# ───────────────────────────────────
# Index access helpers
# ───────────────────────────────────

def get_item_index_df(item):
    rows = (
        ItemIndex.query
        .filter_by(item_id=item.id)
        .order_by(ItemIndex.month.asc())
        .all()
    )
    if not rows:
        return pd.DataFrame()

    data = []
    for r in rows:
        row = {"date": pd.Timestamp(r.month.year, r.month.month, 1)}
        try:
            idx = json.loads(r.indices_json or "{}")
        except Exception:
            idx = {}
        row.update(idx)
        data.append(row)

    df = pd.DataFrame(data)
    if df.empty:
        return df
    return df.set_index("date").sort_index()


def ieema_row(df, dt, previous=False):
    """Return the latest index row on or before the target month."""
    if df is None or df.empty:
        return None
    target = previous_month(dt) if previous else to_month_start(dt)
    if pd.isna(target):
        return None
    eligible = df[df.index <= target]
    return eligible.iloc[-1] if not eligible.empty else None


def pvc_percent(base_date, current_date, idx_df, weights):
    """Return total PVC % given two dates, an index DataFrame, and a weights dict."""
    base = ieema_row(idx_df, base_date, previous=False)
    curr = ieema_row(idx_df, current_date, previous=True)
    if base is None or curr is None:
        return 0.0

    total = 0.0
    for key, weight in weights.items():
        b = base.get(key)
        c = curr.get(key)
        if b not in (None, 0) and c is not None:
            total += float(weight) * (float(c) - float(b)) / float(b)
    return total


def pvc_percent_detailed(base_date, current_date, idx_df, scenario, weights):
    """Return a dict with per-component breakdown of PVC % for one scenario."""
    base = ieema_row(idx_df, base_date, previous=False)
    curr = ieema_row(idx_df, current_date, previous=True)
    if base is None or curr is None:
        return None

    row = {
        "scenario":    scenario,
        "basemonth":   base.name.isoformat() if hasattr(base.name, "isoformat") else str(base.name),
        "currentmonth": curr.name.isoformat() if hasattr(curr.name, "isoformat") else str(curr.name),
        "pvcpercent":  0.0,
    }
    total = 0.0
    for key, weight in weights.items():
        b = base.get(key)
        c = curr.get(key)
        contrib = None
        if b not in (None, 0) and c is not None:
            contrib = round(float(weight) * (float(c) - float(b)) / float(b), 4)
            total += contrib
        row[f"{key}_base"]            = safe_round(b, 2) if b is not None else None
        row[f"{key}_current"]         = safe_round(c, 2) if c is not None else None
        row[f"{key}_weight"]          = weight
        row[f"{key}_contributionpct"] = contrib
    row["pvcpercent"] = round(total, 4)
    return row


# ───────────────────────────────────
# IGBT-specific helpers
# ───────────────────────────────────

def igbt_index_values(idx_df, base_month, ref_month, currency):
    """
    Return (baseidx, curridx) dicts for the IGBT formula.

    Base  : all indices from base_month.
    Current:
      - C / AL / FE / IM / W / D  from  ref_month − 1
      - ER                         from  ref_month − 3
    """
    currency = (currency or "EUR").upper().strip()
    ercol = f"ER_{currency}"

    base_row = ieema_row(idx_df, base_month, previous=False)
    row_1m   = ieema_row(idx_df, previous_month(ref_month, 1), previous=False)
    row_3m   = ieema_row(idx_df, previous_month(ref_month, 3), previous=False)

    if base_row is None or row_1m is None or row_3m is None:
        raise ValueError("IGBT indices missing for one or more required months.")

    def get_val(row, col):
        v = row.get(col)
        if v is None:
            raise ValueError(f"Index column '{col}' is missing in the data.")
        return float(v)

    baseidx = {
        "C":  get_val(base_row, "C"),
        "AL": get_val(base_row, "AL"),
        "FE": get_val(base_row, "FE"),
        "IM": get_val(base_row, "IM"),
        "W":  get_val(base_row, "W"),
        "D":  get_val(base_row, "D"),
        "ER": get_val(base_row, ercol),
    }
    curridx = {
        "C":  get_val(row_1m, "C"),
        "AL": get_val(row_1m, "AL"),
        "FE": get_val(row_1m, "FE"),
        "IM": get_val(row_1m, "IM"),
        "W":  get_val(row_1m, "W"),
        "D":  get_val(row_1m, "D"),
        "ER": get_val(row_3m, ercol),   # ER uses 3-month lag
    }
    return baseidx, curridx


def igbt_p1_indigenous(baseidx, curridx, weights, indigenous_portion):
    """Compute P1 (indigenous component) PVC amount."""
    Fc  = float(weights.get("FIXED", 16))
    wC  = float(weights.get("C",  26))
    wAL = float(weights.get("AL", 13))
    wFE = float(weights.get("FE", 18))
    wIM = float(weights.get("IM",  9))
    wW  = float(weights.get("W",  18))

    factor = (
        Fc
        + wC  * curridx["C"]  / baseidx["C"]
        + wAL * curridx["AL"] / baseidx["AL"]
        + wFE * curridx["FE"] / baseidx["FE"]
        + wIM * curridx["IM"] / baseidx["IM"]
        + wW  * curridx["W"]  / baseidx["W"]
    )
    return indigenous_portion * (factor / 100.0) - indigenous_portion


def igbt_p2_cif(cif, baseidx, curridx):
    """Compute P2 (CIF / imported component) PVC amount."""
    ER0 = baseidx["ER"]
    ER1 = curridx["ER"]
    D0  = baseidx["D"]
    D1  = curridx["D"]
    return cif * (
        (ER1 * (100 + D1) / (100 * ER0))
        - ((100 + D0) / 100)
    )


def igbt_vendor_scenario(rate, vendor, base_month, ref_month, idx_df, weights, scenario_label):
    """
    Compute P1 / P2 and total PVC for one vendor in one scenario.
    Returns (summary_dict, indexdetails_list, total_pvc).

    BUG FIX: The ER index-detail lookup previously used `f"ER_{currency}"` as a key
    into baseidx/curridx, but those dicts store the ER value under the plain "ER" key.
    Now we always look up "ER" in those dicts while still labelling the row correctly.
    """
    currency = (vendor.currency or "EUR").upper()
    baseidx, curridx = igbt_index_values(idx_df, base_month, ref_month, currency)

    cif               = safe_float(vendor.cif)
    baseduty          = cif * baseidx["D"] / 100
    indigenous_portion = rate - cif - baseduty

    p1        = igbt_p1_indigenous(baseidx, curridx, weights, indigenous_portion)
    p2        = igbt_p2_cif(cif, baseidx, curridx)
    total_pvc = p1 + p2

    base_month_str = str(base_month.date()) if hasattr(base_month, "date") else str(base_month)
    curr_month_str = str(ref_month.date())  if hasattr(ref_month,  "date") else str(ref_month)

    summary = {
        "scenario":     scenario_label,
        "vendor":       vendor.vendor_name,
        "pono":         vendor.po_no,
        "currency":     currency,
        "rate":         rate,
        "cif":          cif,
        "baseduty":     baseduty,
        "indigenous":   indigenous_portion,
        "p1":           p1,
        "p2":           p2,
        "totalpvc":     total_pvc,
        "basemonth":    base_month_str,
        "currentmonth": curr_month_str,
    }

    # Index-detail rows (used in result.html IGBT tabs).
    # baseidx/curridx always use the plain key ("ER"); the label carries currency.
    index_columns = [
        ("C",  "Copper Index"),
        ("AL", "Aluminium Index"),
        ("FE", "Iron / Steel Index"),
        ("IM", "Import Machine Index"),
        ("W",  "Labour / Wage Index"),
        ("D",  "Import Duty"),
        ("ER", f"Exchange Rate {currency}"),
    ]
    indexdetails = []
    for col, label in index_columns:
        indexdetails.append({
            "scenario":     scenario_label,
            "vendor":       vendor.vendor_name,
            "parameter":    col if col != "ER" else f"ER_{currency}",
            "label":        label,
            "basemonth":    base_month_str,
            "basevalue":    safe_round(baseidx.get(col), 4),
            "currentmonth": curr_month_str,
            "currentvalue": safe_round(curridx.get(col), 4),
        })

    return summary, indexdetails, total_pvc


# ───────────────────────────────────
# IGBT Propulsion System calculation
# ───────────────────────────────────

def calculate_igbt_propulsion(item, data, tender_id, idx_df, weights):
    """
    Full IGBT Propulsion System PVC calculation.

    Fixes applied vs. original:
      1. `lower_base_date` was undefined — replaced with `lower_base` throughout.
      2. `scheduled_date`  was undefined — replaced with `scheduleddp`.
      3. `pvcd1` was never assigned — now captured from run_scenario("D1", ...).
      4. `qty`  was undefined — replaced with `quantity`.
      5. `lowercontractual` was computed in the special-case block then
         unconditionally overwritten — removed the duplicate assignment.
    """
    tender = TenderMaster.query.get_or_404(int(tender_id))
    vendors = (
        TenderVendor.query
        .filter_by(tender_id=tender.id)
        .order_by(TenderVendor.id.asc())
        .all()
    )
    if not vendors:
        raise ValueError("No vendor rows configured for the selected tender.")

    # ── Merge form data with tender defaults ────────────────────────────────
    basic_rate        = safe_float(data.get("basicrate")           or tender.basicrate)
    pvc_base_date     = data.get("pvcbasedate")                    or tender.pvcbasedate
    lower_rate        = safe_float(data.get("lowerrate")           or tender.lowerrate or 0)
    lower_base        = data.get("lowerbasicdate")                 or tender.lowerratebasedate
    freight_per_unit  = safe_float(data.get("freightrateperunit")  or tender.freightrateperunit or 0)
    lower_freight     = safe_float(data.get("lowerfreight")        or tender.lowerfreight or 0)
    quantity          = safe_float(data.get("quantity")            or 1)
    if quantity <= 0:
        quantity = 1.0

    cal_date      = data.get("caldate")
    origdp        = data.get("origdp")
    refixeddp     = data.get("refixeddp")
    extendeddp    = data.get("extendeddp")
    supply_date   = data.get("supdate")
    rate_applied  = (data.get("rateapplied") or "").strip().lower()
    developmental = bool(data.get("developmentaltender"))

    # "Scheduled DP" — Extended is only for LD. PVC uses Refixed or Original.
    pvc_scheduleddp = refixeddp or origdp
    ld_scheduleddp  = extendeddp or refixeddp or origdp

    schedts  = pd.to_datetime(ld_scheduleddp,  errors="coerce")
    supplyts = pd.to_datetime(supply_date,   errors="coerce")

    basemonth        = to_month_start(pvc_base_date)
    lowerbase_month  = to_month_start(lower_base) if lower_base else None
    calmonth         = to_month_start(cal_date)
    schedmonth       = to_month_start(pvc_scheduleddp)

    lowerrate_applicable = lower_rate > 0 and lowerbase_month is not None

    freight_total       = freight_per_unit * quantity
    lowerfreight_total  = lower_freight    * quantity

    # ── LD calculation ──────────────────────────────────────────────────────
    delaydays    = 0
    ldweeks      = 0
    ldratepct    = 0.0
    ldapplicable = False
    if pd.notna(schedts) and pd.notna(supplyts) and supplyts > schedts:
        delaydays    = int((supplyts - schedts).days)
        ldweeks      = math.ceil(delaydays / 7) if delaydays > 0 else 0
        ldratepct    = min(ldweeks * 0.5, 10.0)
        ldapplicable = True

    if developmental:
        ldapplicable = False
        ldratepct    = 0.0
        ldweeks      = 0

    # ── D1 special-case check FIRST ────────
    d1_special_case  = False
    _d1_plain_igbt   = None
    if lowerrate_applicable:
        lower_base_ts = pd.to_datetime(lower_base, errors="coerce")
        origdp_ts     = pd.to_datetime(origdp,     errors="coerce")
        refix_ts      = pd.to_datetime(refixeddp,  errors="coerce")

        # Prefer Refixed DP; fall back to Original DP
        compare_date = refix_ts if pd.notna(refix_ts) else origdp_ts

        if pd.notna(lower_base_ts) and pd.notna(compare_date) and lower_base_ts > compare_date:
            d1_special_case = True
            _d1_plain_igbt  = (lower_rate * quantity + lowerfreight_total) * GST_FACTOR

    # ── Run scenarios (returns min PVC per set across vendors) ───────────────
    all_vendor_summaries: list = []
    all_index_details:    list = []

    def run_scenario(label, rate, base_m, ref_m):
        if not base_m or pd.isna(base_m):
            return None
        if not ref_m or pd.isna(ref_m):
            return None
        sc_summaries, sc_indices = [], []
        for v in vendors:
            try:
                s, idxrows, _ = igbt_vendor_scenario(
                    rate, v, base_m, ref_m, idx_df, weights, label
                )
                sc_summaries.append(s)
                sc_indices.extend(idxrows)
            except ValueError:
                continue
        all_vendor_summaries.extend(sc_summaries)
        all_index_details.extend(sc_indices)
        if not sc_summaries:
            return None
        valid = [s["totalpvc"] for s in sc_summaries if s["totalpvc"] is not None]
        return min(valid) if valid else None

    pvca2 = run_scenario("A2", basic_rate, basemonth,       calmonth)
    pvcb2 = run_scenario("B2", basic_rate, basemonth,       schedmonth)
    pvcc1 = run_scenario("C1", lower_rate, lowerbase_month, calmonth)   if lowerrate_applicable else None
    pvcd1 = None
    if lowerrate_applicable and not d1_special_case:
        pvcd1 = run_scenario("D1", lower_rate, lowerbase_month, schedmonth)

    # ── Amount calculation helper ────────────────────────────────────────────
    def with_freight_gst(base_rate, pvc_per_set, qty, frt):
        """(base_rate × qty + pvc_per_set × qty + freight) × GST"""
        if pvc_per_set is None:
            return None
        return (base_rate * qty + pvc_per_set * qty + frt) * GST_FACTOR

    pvcactual      = with_freight_gst(basic_rate, pvca2, quantity, freight_total)
    pvccontractual = with_freight_gst(basic_rate, pvcb2, quantity, freight_total)

    # ── LD deductions ────────────────────────────────────────────────────────
    ldamtactual      = (max(pvcactual,      0) * (ldratepct / 100.0)) if (ldapplicable and pvcactual      is not None) else 0.0
    ldamtcontractual = (max(pvccontractual, 0) * (ldratepct / 100.0)) if (ldapplicable and pvccontractual is not None) else 0.0

    pvcactuallessld      = (pvcactual      - ldamtactual)      if pvcactual      is not None else None
    pvccontractuallessld = (pvccontractual - ldamtcontractual) if pvccontractual is not None else None

    loweractual      = None
    lowercontractual = None
    if lowerrate_applicable:
        loweractual = with_freight_gst(lower_rate, pvcc1, quantity, lowerfreight_total)

        if d1_special_case:
            # D1 special case uses B2 net (after LD)
            b2_for_comparison = pvccontractuallessld if pvccontractuallessld is not None else pvccontractual
            lowercontractual = min(b2_for_comparison, _d1_plain_igbt) if b2_for_comparison is not None else _d1_plain_igbt
        else:
            lowercontractual = with_freight_gst(lower_rate, pvcd1, quantity, lowerfreight_total)

    loweractuallessld = (loweractual - ldamtactual) if loweractual is not None else None
    if d1_special_case:
        lowercontractuallessld = lowercontractual
    else:
        lowercontractuallessld = (lowercontractual - ldamtcontractual) if lowercontractual is not None else None

    # ── Scenario selection ───────────────────────────────────────────────────
    candidates = _build_candidates(
        rate_applied,
        pvcactual, pvccontractual,
        pvcactuallessld, pvccontractuallessld,
        loweractual, lowercontractual,
        loweractuallessld, lowercontractuallessld,
        lowerrate_applicable,
    )
    candidates  = {k: v for k, v in candidates.items() if v is not None}
    selected    = min(candidates, key=candidates.get) if candidates else "A2"
    fairprice   = candidates.get(selected, pvcactual)

    scenarioamounts = {"A2": safe_round(pvcactuallessld), "B2": safe_round(pvccontractuallessld)}
    if lowerrate_applicable:
        scenarioamounts["C1"] = safe_round(loweractual)
        scenarioamounts["D1"] = safe_round(lowercontractual)

    # ── Scenario index details ───────────────────────────────────────────────
    scenariodetails: list = []
    ieema_scenarios = [
        ("A2", pvc_base_date, cal_date),
        ("B2", pvc_base_date, pvc_scheduleddp),
    ]
    if lowerrate_applicable:
        ieema_scenarios.append(("C1", lower_base, cal_date))
        # D1 special case: no PVC% computed, so no index detail row for D1.
        if not d1_special_case:
            ieema_scenarios.append(("D1", lower_base, pvc_scheduleddp))
    for sc, bd, cd in ieema_scenarios:
        if bd and cd:
            det = pvc_percent_detailed(bd, cd, idx_df, sc, weights)
            if det:
                scenariodetails.append(det)

    return {
        "pvcactual":               safe_round(pvcactual),
        "pvccontractual":          safe_round(pvccontractual),
        "loweractual":             safe_round(loweractual)            if lowerrate_applicable else None,
        "lowercontractual":        safe_round(lowercontractual)       if lowerrate_applicable else None,
        "delaydays":               delaydays,
        "ldweeks":                 ldweeks,
        "ldratepct":               safe_round(ldratepct),
        "ldapplicable":            ldapplicable,
        "ldamtactual":             safe_round(ldamtactual),
        "ldamtcontractual":        safe_round(ldamtcontractual),
        "pvcactuallessldnew":      safe_round(pvcactuallessld),
        "pvccontractuallessldnew": safe_round(pvccontractuallessld),
        "loweractuallessld":       safe_round(loweractuallessld)      if loweractuallessld      is not None else None,
        "lowercontractuallessld":  safe_round(lowercontractuallessld) if lowercontractuallessld is not None else None,
        "fairpricenew":            safe_round(fairprice),
        "selectedscenarionew":     selected,
        "pvcperseta2":             safe_round(pvca2),
        "pvcpersetb2":             safe_round(pvcb2),
        "pvcpersetc1":             safe_round(pvcc1) if pvcc1 is not None else None,
        # D1 special case: pvcd1 is None (no vendor PVC calc), so pvcpersetd1 is None.
        "pvcpersetd1":             (None if d1_special_case else (safe_round(pvcd1) if pvcd1 is not None else None)),
        "scenarioamounts":         scenarioamounts,
        "scenariodetails":         scenariodetails,
        "igbtvendordetails":       all_vendor_summaries,
        "igbtindexdetails":        all_index_details,
        "tenderno":                tender.tender_no,
        "pono":                    vendors[0].po_no if vendors else None,
    }


# ───────────────────────────────────
# IEEMA single-record calculation
# ───────────────────────────────────

def calc_single_record(data, idx_df, weights):
    """
    Standard IEEMA PVC calculation (non-IGBT items).

    Fix: `lowercontractual` was computed correctly inside the D1 special-case
    block but then unconditionally overwritten by a plain formula a few lines
    below.  The plain formula now only runs in the normal (else) branch.
    """
    pvc_base_date   = data.get("pvcbasedate")
    cal_date        = data.get("caldate")
    origdp          = data.get("origdp")
    refixeddp       = data.get("refixeddp")
    extendeddp      = data.get("extendeddp")
    supply_date     = data.get("supdate")
    lower_base_date = data.get("lowerbasicdate")

    pvc_scheduled_date = refixeddp or origdp
    ld_scheduled_date  = extendeddp or refixeddp or origdp

    qty               = safe_float(data.get("quantity"))
    if qty <= 0:
        qty = 1.0
    basic_rate        = safe_float(data.get("basicrate"))
    freight_per_unit  = safe_float(data.get("freightrateperunit"))
    lower_rate        = safe_float(data.get("lowerrate"))
    lower_freight     = safe_float(data.get("lowerfreight"))
    developmental     = bool(data.get("developmentaltender"))

    freight_total      = freight_per_unit * qty
    lowerfreight_total = lower_freight    * qty

    lowerrate_applicable = lower_rate > 0 and bool(lower_base_date)

    # ── Scenario percentages ────────────────────────────────────────────────
    pct_a2 = pvc_percent(pvc_base_date,   cal_date,       idx_df, weights)
    pct_b2 = pvc_percent(pvc_base_date,   pvc_scheduled_date, idx_df, weights)
    pct_c1 = pvc_percent(lower_base_date, cal_date,       idx_df, weights) if lowerrate_applicable else None

    pct_d1           = None
    lowercontractual = None   # will be set in either branch below
    d1_special_case  = False  # flag used later to suppress pvcpersetd1

    if lowerrate_applicable:
        lower_base_ts = pd.to_datetime(lower_base_date, errors="coerce")
        origdp_ts     = pd.to_datetime(origdp,          errors="coerce")
        refix_ts      = pd.to_datetime(refixeddp,       errors="coerce")

        # Prefer Refixed DP; fall back to Original DP
        compare_date = refix_ts if pd.notna(refix_ts) else origdp_ts

        if pd.notna(lower_base_ts) and pd.notna(compare_date) and lower_base_ts > compare_date:
            # ── Special case: Lower Base Date > reference DP ─────────────────
            # No PVC% is calculated for D1.  Instead D1 is the minimum of:
            #   • B2 amount (basic_rate with B2 PVC, already computed above)
            #   • Plain lower basic amount (lower_rate × qty + freight, no PVC)
            # pvccontractual (B2) is computed after this block, so we store a
            # lambda and resolve it once pvccontractual is available.
            d1_special_case  = True
            pct_d1           = None   # no PVC% for D1 in this branch
            # plain lower basic amount (no PVC)
            d1_plain = (lower_rate * qty + lowerfreight_total) * GST_FACTOR
            # lowercontractual is set after pvccontractual is known (see below)
            _d1_plain_amount = d1_plain   # stash for use after pvccontractual

        else:
            # ── Normal case: D1 = lower_rate PVC from Lower Base Date → Scheduled Date
            pct_d1 = pvc_percent(lower_base_date, pvc_scheduled_date, idx_df, weights)
            lowercontractual = (lower_rate * qty * (1 + (pct_d1 or 0) / 100.0) + lowerfreight_total) * GST_FACTOR

    # ── Gross amounts & LD ───────────────────────────────────────────────────
    pvcactual = (
        basic_rate * qty * (1 + pct_a2 / 100.0) + freight_total
    ) * GST_FACTOR

    pvccontractual = (
        basic_rate * qty * (1 + pct_b2 / 100.0) + freight_total
    ) * GST_FACTOR

    delaydays    = 0
    ldweeks      = 0
    ldratepct    = 0.0
    ldapplicable = False
    duets    = pd.to_datetime(ld_scheduled_date, errors="coerce")
    supplyts = pd.to_datetime(supply_date,    errors="coerce")
    if pd.notna(duets) and pd.notna(supplyts) and supplyts > duets:
        delaydays    = int((supplyts - duets).days)
        ldweeks      = math.ceil(delaydays / 7) if delaydays > 0 else 0
        ldratepct    = min(ldweeks * 0.5, 10.0)
        ldapplicable = True

    if developmental:
        ldapplicable = False
        ldratepct    = 0.0
        ldweeks      = 0

    ldamtactual      = pvcactual      * (ldratepct / 100.0) if ldapplicable else 0.0
    ldamtcontractual = pvccontractual * (ldratepct / 100.0) if ldapplicable else 0.0

    pvcactuallessld      = pvcactual      - ldamtactual
    pvccontractuallessld = pvccontractual - ldamtcontractual

    loweractual = None
    if lowerrate_applicable:
        loweractual = (
            lower_rate * qty * (1 + (pct_c1 or 0) / 100.0) + lowerfreight_total
        ) * GST_FACTOR

        if d1_special_case:
            # D1 special case uses B2 net (after LD)
            lowercontractual = min(pvccontractuallessld, _d1_plain_amount)

    loweractuallessld = (loweractual - ldamtactual) if loweractual is not None else None
    if d1_special_case:
        lowercontractuallessld = lowercontractual
    else:
        lowercontractuallessld = (lowercontractual - ldamtcontractual) if lowercontractual is not None else None

    # ── Scenario selection ───────────────────────────────────────────────────
    candidates = _build_candidates(
        (data.get("rateapplied") or "").strip().lower(),
        pvcactual, pvccontractual,
        pvcactuallessld, pvccontractuallessld,
        loweractual, lowercontractual,
        loweractuallessld, lowercontractuallessld,
        lowerrate_applicable,
    )
    candidates  = {k: v for k, v in candidates.items() if v is not None}
    selected    = min(candidates, key=candidates.get) if candidates else "A2"
    fairprice   = candidates.get(selected, pvcactual)

    # FIX: scenarioamounts now consistently uses the less-LD values for A2/B2
    # and gross values for C1/D1 (matching the original intent — LD is only
    # deducted from the basic-rate scenarios A2/B2, not from lower-rate ones).
    scenarioamounts = {
        "A2": safe_round(pvcactuallessld),
        "B2": safe_round(pvccontractuallessld),
    }
    if lowerrate_applicable:
        scenarioamounts["C1"] = safe_round(loweractual)
        scenarioamounts["D1"] = safe_round(lowercontractual)

    scenariodetails: list = []
    _sc_list = [
        ("A2", pvc_base_date,   cal_date),
        ("B2", pvc_base_date,   pvc_scheduled_date),
        ("C1", lower_base_date, cal_date),
    ]
    # D1 special case: no PVC% was computed, so no index detail row for D1.
    if lowerrate_applicable and not d1_special_case:
        _sc_list.append(("D1", lower_base_date, pvc_scheduled_date))
    for sc, bd, cd in _sc_list:
        if bd and cd:
            det = pvc_percent_detailed(bd, cd, idx_df, sc, weights)
            if det:
                scenariodetails.append(det)

    return {
        "pvcactual":               safe_round(pvcactual),
        "pvccontractual":          safe_round(pvccontractual),
        "loweractual":             safe_round(loweractual)      if loweractual      is not None else None,
        "lowercontractual":        safe_round(lowercontractual) if lowercontractual is not None else None,
        "delaydays":               delaydays,
        "ldweeks":                 ldweeks,
        "ldratepct":               safe_round(ldratepct),
        "ldapplicable":            ldapplicable,
        "ldamtactual":             safe_round(ldamtactual),
        "ldamtcontractual":        safe_round(ldamtcontractual),
        "pvcactuallessldnew":      safe_round(pvcactuallessld),
        "pvccontractuallessldnew": safe_round(pvccontractuallessld),
        "loweractuallessld":       safe_round(loweractuallessld)      if loweractuallessld      is not None else None,
        "lowercontractuallessld":  safe_round(lowercontractuallessld) if lowercontractuallessld is not None else None,
        "fairpricenew":            safe_round(fairprice),
        "selectedscenarionew":     selected,
        "pvcperseta2":             safe_round(basic_rate * pct_a2 / 100.0),
        "pvcpersetb2":             safe_round(basic_rate * pct_b2 / 100.0),
        "pvcpersetc1":             safe_round(lower_rate * (pct_c1 or 0) / 100.0) if lowerrate_applicable else None,
        # D1 special case: no PVC% calculated, so pvcpersetd1 is None.
        "pvcpersetd1":             (None if (not lowerrate_applicable or d1_special_case)
                                    else safe_round(lower_rate * (pct_d1 or 0) / 100.0)),
        "scenarioamounts":         scenarioamounts,
        "scenariodetails":         scenariodetails,
        "igbtvendordetails":       [],
        "igbtindexdetails":        [],
        "tenderno":                None,
        "pono":                    None,
    }


# ───────────────────────────────────
# Shared scenario-selection helper
# ───────────────────────────────────

def _build_candidates(
    rate_applied,
    pvcactual, pvccontractual,
    pvcactuallessld, pvccontractuallessld,
    loweractual, lowercontractual,
    loweractuallessld, lowercontractuallessld,
    lowerrate_applicable,
):
    """
    Return the candidate dict for fair-price selection based on `rate_applied`.
    Extracted to avoid duplication between calc_single_record and
    calculate_igbt_propulsion.
    """
    if rate_applied == "supply before due date":
        return {"A2": pvcactual}

    if rate_applied == "supply after due date":
        return {"A2": pvcactuallessld, "B2": pvccontractuallessld}

    if rate_applied == "lower rate applicable":
        d = {"A2": pvcactual, "B2": pvccontractual}
        if lowerrate_applicable:
            d["C1"] = loweractual
            d["D1"] = lowercontractual
        return d

    if rate_applied == "lower rate and ld comparison":
        d = {"A2": pvcactuallessld, "B2": pvccontractuallessld}
        if lowerrate_applicable:
            d["C1"] = loweractual
            d["D1"] = lowercontractual
        return d

    if rate_applied == "lower rate with ld in further extension":
        d = {"A2": pvcactuallessld, "B2": pvccontractuallessld}
        if lowerrate_applicable:
            d["C1"] = loweractuallessld
            d["D1"] = lowercontractuallessld
        return d

    # Default: all applicable scenarios
    d = {"A2": pvcactual, "B2": pvccontractual}
    if lowerrate_applicable:
        d["C1"] = loweractual
        d["D1"] = lowercontractual
    return d


# ───────────────────────────────────
# Dispatcher
# ───────────────────────────────────

def calculate_for_item(item, data, idx_df, weights):
    code = (item.pvc_formula_code or "").replace(" ", "").upper()
    if code == "IGBTPROPULSIONSYSTEM":
        tender_id = data.get("tenderid")
        if not tender_id:
            raise ValueError("Tender is required for IGBT Propulsion System.")
        return calculate_igbt_propulsion(item, data, tender_id, idx_df, weights)
    return calc_single_record(data, idx_df, weights)


# ───────────────────────────────────
# PVC input + result wrappers
# ───────────────────────────────────

class PVCInput:
    _KNOWN = {
        "itemid", "item_id", "basicrate", "quantity", "freightrateperunit",
        "pvcbasedate", "origdp", "refixeddp", "extendeddp", "caldate", "supdate",
        "lowerrate", "lowerfreight", "lowerbasicdate", "rateapplied",
        "tenderid", "tender_id", "developmentaltender",
    }

    def __init__(self, form):
        self.user_id               = None
        self.username              = None
        self.item_id               = int(form.get("itemid") or form.get("item_id") or 0)
        self.basic_rate            = safe_float(form.get("basicrate"))
        self.quantity              = safe_float(form.get("quantity"))
        self.freight_rate_per_unit = safe_float(form.get("freightrateperunit"))
        self.pvc_base_date         = form.get("pvcbasedate")
        self.original_dp           = form.get("origdp")
        self.refixed_dp            = form.get("refixeddp")
        self.extended_dp           = form.get("extendeddp")
        self.cal_date              = form.get("caldate")
        self.supply_date           = form.get("supdate")
        self.lower_rate            = safe_float(form.get("lowerrate"))
        self.lower_freight         = safe_float(form.get("lowerfreight"))
        self.lower_basic_date      = form.get("lowerbasicdate")
        self.rate_applied          = form.get("rateapplied")
        self.tender_id             = form.get("tenderid") or form.get("tender_id")
        self.developmental_tender  = bool(form.get("developmentaltender"))
        self.extra_data = {
            k: form.get(k)
            for k in form.keys()
            if k not in self._KNOWN
        }

    def to_dict(self):
        d = {
            "basicrate":           self.basic_rate,
            "quantity":            self.quantity,
            "freightrateperunit":  self.freight_rate_per_unit,
            "pvcbasedate":         self.pvc_base_date,
            "origdp":              self.original_dp,
            "refixeddp":           self.refixed_dp,
            "extendeddp":          self.extended_dp,
            "caldate":             self.cal_date,
            "supdate":             self.supply_date,
            "lowerrate":           self.lower_rate,
            "lowerfreight":        self.lower_freight,
            "lowerbasicdate":      self.lower_basic_date,
            "rateapplied":         self.rate_applied,
            "tenderid":            self.tender_id,
            "developmentaltender": self.developmental_tender,
        }
        d.update(self.extra_data)
        return d


class ResultObj:
    def __init__(self, payload):
        self.data              = payload
        self.scenarioamounts   = payload.get("scenarioamounts",   {})
        self.scenariodetails   = payload.get("scenariodetails",   [])
        self.igbtvendordetails = payload.get("igbtvendordetails", [])
        self.igbtindexdetails  = payload.get("igbtindexdetails",  [])


# ───────────────────────────────────
# Routes
# ───────────────────────────────────

@pvc_bp.route("/calculate", methods=["POST"])
@login_required
def calculate():
    itemid = request.form.get("itemid") or request.form.get("item_id")
    if not itemid:
        flash("Please select an item.", "danger")
        return redirect(url_for("main.index"))

    item   = Item.query.get_or_404(int(itemid))
    idx_df = get_item_index_df(item)
    if idx_df.empty:
        flash(
            "Indices not configured for this item. "
            "Please add them via Admin → Item Indices.",
            "danger",
        )
        return redirect(url_for("main.index"))

    try:
        weights = json.loads(item.weights_json or "{}")
    except Exception:
        weights = {}

    pvc_input          = PVCInput(request.form)
    pvc_input.user_id  = current_user.id
    pvc_input.username = current_user.username
    data = pvc_input.to_dict()

    try:
        result_row = calculate_for_item(item, data, idx_df, weights)
    except ValueError as e:
        flash(str(e), "danger")
        return redirect(url_for("main.index"))
    except Exception:
        flash("Calculation failed. Please check your inputs and index data.", "danger")
        return redirect(url_for("main.index"))

    payload = {
        "pvcactual":               result_row.get("pvcactual",               0.0),
        "pvccontractual":          result_row.get("pvccontractual",          0.0),
        "loweractual":             result_row.get("loweractual"),
        "lowercontractual":        result_row.get("lowercontractual"),
        "ldamtactual":             result_row.get("ldamtactual",             0.0),
        "ldamtcontractual":        result_row.get("ldamtcontractual",        0.0),
        "fairprice":               result_row.get("fairpricenew",            0.0),
        "pvcactuallessldnew":      result_row.get("pvcactuallessldnew"),
        "pvccontractuallessldnew": result_row.get("pvccontractuallessldnew"),
        "loweractuallessld":       result_row.get("loweractuallessld"),
        "lowercontractuallessld":  result_row.get("lowercontractuallessld"),
        "delaydays":               result_row.get("delaydays",               0),
        "ldweeks":                 result_row.get("ldweeks",                 0),
        "ldratepct":               result_row.get("ldratepct",               0.0),
        "ldapplicable":            result_row.get("ldapplicable",            False),
        "selectedscenario":        result_row.get("selectedscenarionew"),
        "pvcperseta2":             result_row.get("pvcperseta2"),
        "pvcpersetb2":             result_row.get("pvcpersetb2"),
        "pvcpersetc1":             result_row.get("pvcpersetc1"),
        "pvcpersetd1":             result_row.get("pvcpersetd1"),
        "tenderno":                result_row.get("tenderno"),
        "pono":                    result_row.get("pono"),
        "scenarioamounts":         result_row.get("scenarioamounts",         {}),
        "scenariodetails":         result_row.get("scenariodetails",         []),
        "igbtvendordetails":       result_row.get("igbtvendordetails",       []),
        "igbtindexdetails":        result_row.get("igbtindexdetails",        []),
        "developmentaltender":     data.get("developmentaltender",           False),
    }

    # Sanitize NaN / Inf → None before touching the DB
    payload = {k: nan_to_none(v) for k, v in payload.items()}

    # JSON-encode list/dict sub-fields and sanitize NaN literals within them
    scenarioamounts_json     = nan_to_none(json.dumps(payload["scenarioamounts"]))
    scenariodetails_json     = nan_to_none(json.dumps(payload["scenariodetails"]))
    igbt_vendor_details_json = nan_to_none(json.dumps(payload["igbtvendordetails"]))

    calc = PVCResult(
        user_id=current_user.id,
        username=current_user.username,
        item_id=item.id,
        basicrate=data.get("basicrate",              0),
        quantity=data.get("quantity",                0),
        freightrateperunit=data.get("freightrateperunit", 0),
        pvcbasedate=data.get("pvcbasedate"),
        origdp=data.get("origdp"),
        refixeddp=data.get("refixeddp"),
        extendeddp=data.get("extendeddp"),
        caldate=data.get("caldate"),
        supdate=data.get("supdate"),
        rateapplied=data.get("rateapplied"),
        pvcactual=payload["pvcactual"],
        pvccontractual=payload["pvccontractual"],
        loweractual=payload["loweractual"],
        lowercontractual=payload["lowercontractual"],
        ldamtactual=payload["ldamtactual"],
        ldamtcontractual=payload["ldamtcontractual"],
        fairprice=payload["fairprice"],
        selectedscenario=payload["selectedscenario"],
        pvcactuallessldnew=payload["pvcactuallessldnew"],
        pvccontractuallessldnew=payload["pvccontractuallessldnew"],
        loweractuallessld=payload["loweractuallessld"],
        lowercontractuallessld=payload["lowercontractuallessld"],
        delaydays=payload["delaydays"],
        ldweeksnew=payload["ldweeks"],
        ldratepctnew=payload["ldratepct"],
        ldapplicable=payload["ldapplicable"],
        pvcperseta2=payload["pvcperseta2"],
        pvcpersetb2=payload["pvcpersetb2"],
        pvcpersetc1=payload["pvcpersetc1"],
        pvcpersetd1=payload["pvcpersetd1"],
        tenderno=payload["tenderno"],
        pono=payload["pono"],
        scenarioamounts_json=scenarioamounts_json,
        scenariodetails_json=scenariodetails_json,
        igbt_vendor_details_json=igbt_vendor_details_json,
    )
    db.session.add(calc)
    db.session.commit()

    result = ResultObj(payload)
    return render_template(
        "result.html",
        item=item.name,
        itemobj=item,
        data=data,
        result=result,
        calcid=calc.id,
    )


@pvc_bp.route("/calc/<int:calcid>")
@login_required
def view_calc(calcid):
    calc = PVCResult.query.filter_by(id=calcid, user_id=current_user.id).first_or_404()
    item = calc.item

    # NOTE: PVCResult does not persist lower_rate or lower_freight as columns.
    # We reconstruct what we can; display templates should handle None gracefully.
    data = {
        "basicrate":           calc.basicrate,
        "quantity":            calc.quantity,
        "freightrateperunit":  calc.freightrateperunit,
        "pvcbasedate":         calc.pvcbasedate,
        "origdp":              calc.origdp,
        "refixeddp":           calc.refixeddp,
        "extendeddp":          calc.extendeddp,
        "caldate":             calc.caldate,
        "supdate":             calc.supdate,
        "rateapplied":         calc.rateapplied,
        "lowerrate":           None,   # not stored; templates must tolerate None
        "lowerfreight":        0,
        "lowerbasicdate":      None,
        "developmentaltender": False,
    }

    payload = {
        "pvcactual":               calc.pvcactual,
        "pvccontractual":          calc.pvccontractual,
        "loweractual":             calc.loweractual,
        "lowercontractual":        calc.lowercontractual,
        "ldamtactual":             calc.ldamtactual,
        "ldamtcontractual":        calc.ldamtcontractual,
        "fairprice":               calc.fairprice,
        "selectedscenario":        calc.selectedscenario,
        "ldapplicable":            calc.ldapplicable,
        "pvcactuallessldnew":      calc.pvcactuallessldnew,
        "pvccontractuallessldnew": calc.pvccontractuallessldnew,
        "loweractuallessld":       calc.loweractuallessld,
        "lowercontractuallessld":  calc.lowercontractuallessld,
        "delaydays":               calc.delaydays,
        "ldweeks":                 calc.ldweeksnew,
        "ldratepct":               calc.ldratepctnew,
        "pvcperseta2":             calc.pvcperseta2,
        "pvcpersetb2":             calc.pvcpersetb2,
        "pvcpersetc1":             calc.pvcpersetc1,
        "pvcpersetd1":             calc.pvcpersetd1,
        "tenderno":                calc.tenderno,
        "pono":                    calc.pono,
        "scenarioamounts":         json.loads(calc.scenarioamounts_json     or "{}"),
        "scenariodetails":         json.loads(calc.scenariodetails_json     or "[]"),
        "igbtvendordetails":       json.loads(calc.igbt_vendor_details_json or "[]"),
        "igbtindexdetails":        [],   # not persisted; only available at calculation time
        "developmentaltender":     False,
    }

    result = ResultObj(payload)
    return render_template(
        "result.html",
        item=item.name if item else "",
        itemobj=item,
        data=data,
        result=result,
        calcid=calc.id,
    )


@pvc_bp.route("/calc/<int:calcid>/excel")
@login_required
def export_calc_excel(calcid):
    calc = PVCResult.query.filter_by(id=calcid, user_id=current_user.id).first_or_404()
    item = calc.item

    maindata = {
        "Calc ID":            calc.id,
        "User":               calc.username,
        "Item":               item.name if item else "",
        "Basic Rate":         calc.basicrate,
        "Quantity":           calc.quantity,
        "FreightUnit":        calc.freightrateperunit,
        "PVC Base Date":      calc.pvcbasedate,
        "Original DP":        calc.origdp,
        "Refixed DP":         calc.refixeddp,
        "Extended DP":        calc.extendeddp,
        "Call Date":          calc.caldate,
        "Supply Date":        calc.supdate,
        "Rate Applied":       calc.rateapplied,
        "PVC Actual":         calc.pvcactual,
        "PVC Contractual":    calc.pvccontractual,
        "Lower Actual":       calc.loweractual,
        "Lower Contractual":  calc.lowercontractual,
        "LD Days":            calc.delaydays,
        "LD Weeks":           calc.ldweeksnew,
        "LD Rate":            calc.ldratepctnew,
        "LD Amt Actual":      calc.ldamtactual,
        "LD Amt Contractual": calc.ldamtcontractual,
        "PVC Actual-LD":      calc.pvcactuallessldnew,
        "PVC Contractual-LD": calc.pvccontractuallessldnew,
        "Lower Actual-LD":    calc.loweractuallessld,
        "Lower Contr-LD":     calc.lowercontractuallessld,
        "Fair Price":         calc.fairprice,
        "Selected Scenario":  calc.selectedscenario,
        "Tender No":          calc.tenderno,
        "PO No":              calc.pono,
    }

    sadata = [
        {"Scenario": sc, "Amount": amount}
        for sc, amount in (json.loads(calc.scenarioamounts_json or "{}") or {}).items()
    ]

    sddata = []
    for det in json.loads(calc.scenariodetails_json or "[]"):
        row = {
            "Scenario":      det.get("scenario"),
            "Base Month":    det.get("basemonth"),
            "Current Month": det.get("currentmonth"),
            "PVC":           det.get("pvcpercent"),
        }
        for k, v in det.items():
            if k not in ("scenario", "basemonth", "currentmonth", "pvcpercent"):
                row[k] = v
        sddata.append(row)

    igbtvendors = json.loads(calc.igbt_vendor_details_json or "[]")

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([maindata]).to_excel(writer, index=False, sheet_name="PVC Result")
        if sadata:
            pd.DataFrame(sadata).to_excel(writer, index=False, sheet_name="Scenario Amounts")
        if sddata:
            pd.DataFrame(sddata).to_excel(writer, index=False, sheet_name="Index Details")
        if igbtvendors:
            pd.DataFrame(igbtvendors).to_excel(writer, index=False, sheet_name="IGBT Vendors")

    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name=f"PVC_Calc_{calc.id}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@pvc_bp.route("/gettender/<int:tenderid>")
@login_required
def get_tender(tenderid):
    tender = TenderMaster.query.get_or_404(tenderid)
    return jsonify(
        basicrate=tender.basicrate,
        freightrateperunit=tender.freightrateperunit,
        pvcbasedate=tender.pvcbasedate,
        lowerrate=tender.lowerrate,
        lowerfreight=tender.lowerfreight,
        lowerbasicdate=tender.lowerratebasedate,
        itemid=tender.item_id,
        tenderno=tender.tender_no,
    )
