# Streamlit Slotting Map App
# ------------------------------------------------------------
# Features (Phase 1):
# 1) Upload a CSV with columns for Lat, Long, Time, Operator, Signal, BTS ID (names may vary)
#    — UI lets you map your CSV columns to these fields via dropdowns.
# 2) Sort by Time, plot points on a map, and compute total path distance (sum of pairwise geodesic distances).
# 3) "Slotting": Given a Slot Length (SL, in meters), divide the full path into consecutive slots.
#    Draw rectangles of size (length = SL, width = user-selected — default 5 m) that tile the path. Rectangles are oriented along the local path.
#
# How to run locally:
#   pip install streamlit pandas numpy pydeck shapely pyproj
#   streamlit run app.py
# ------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import json
import pandas as pd
import pydeck as pdk
import streamlit as st
from shapely.geometry import LineString, Point, Polygon
from pyproj import Transformer

# ------------------------
# Helpers
# ------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance between two (lat, lon) points in meters."""
    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    # Haversine formula
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    R = 6371000.0  # mean Earth radius in meters
    return R * c


def total_path_distance_m(lats: np.ndarray, lons: np.ndarray) -> float:
    if len(lats) < 2:
        return 0.0
    d = 0.0
    for i in range(1, len(lats)):
        d += haversine_m(lats[i-1], lons[i-1], lats[i], lons[i])
    return d


def utm_epsg_for_latlon(lat: float, lon: float) -> int:
    """Return EPSG code for an appropriate UTM zone for the given lat/lon."""
    zone = int(math.floor((lon + 180) / 6) + 1)
    if lat >= 0:
        return 32600 + zone  # WGS84 / UTM Northern Hemisphere
    else:
        return 32700 + zone  # WGS84 / UTM Southern Hemisphere


def build_projectors(lat: float, lon: float):
    epsg = utm_epsg_for_latlon(lat, lon)
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)  # lon, lat -> x, y
    inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)  # x, y -> lon, lat
    return fwd, inv


def to_xy(fwd: Transformer, coords_lon_lat: List[Tuple[float, float]]):
    xs, ys = [], []
    for lon, lat in coords_lon_lat:
        x, y = fwd.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def to_lonlat(inv: Transformer, coords_xy: List[Tuple[float, float]]):
    lons, lats = [], []
    for x, y in coords_xy:
        lon, lat = inv.transform(x, y)
        lons.append(lon)
        lats.append(lat)
    return list(zip(lons, lats))


@dataclass
class SlotRectangle:
    index: int
    polygon_lonlat: List[Tuple[float, float]]  # list of (lon, lat) corners in order
    start_m: float
    end_m: float


def generate_slot_rectangles(
    lats: np.ndarray,
    lons: np.ndarray,
    slot_len_m: float,
    width_m: float = 1.0,
) -> List[SlotRectangle]:
    """Given a lat/lon path, create consecutive rectangles (length=slot_len_m, width=width_m)
    that tile the path. Rectangles are oriented along the local path direction.
    Returns list of SlotRectangle with polygon coords in lon/lat.
    """
    if len(lats) < 2:
        return []

    # Use an appropriate local projection so meters make sense
    mid_lat = float(np.nanmean(lats))
    mid_lon = float(np.nanmean(lons))
    fwd, inv = build_projectors(mid_lat, mid_lon)

    # Build LineString in projected (x, y) meters
    coords_lonlat = list(zip(lons.tolist(), lats.tolist()))  # (lon, lat)
    xs, ys = to_xy(fwd, coords_lonlat)
    line = LineString(list(zip(xs, ys)))

    total_len = line.length
    if total_len == 0:
        return []

    n_slots = int(math.ceil(total_len / float(slot_len_m)))
    rects: List[SlotRectangle] = []

    half_w = width_m / 2.0

    for i in range(n_slots):
        start_d = i * slot_len_m
        end_d = min((i + 1) * slot_len_m, total_len)

        # Start & end points along the path in projected coords
        p0: Point = line.interpolate(start_d)
        p1: Point = line.interpolate(end_d)

        dx = p1.x - p0.x
        dy = p1.y - p0.y
        seg_len = math.hypot(dx, dy)
        if seg_len == 0:
            # Degenerate slot (can happen if repeated points), skip
            continue

        # Unit direction vector along slot
        ux, uy = dx / seg_len, dy / seg_len
        # Perpendicular unit vector (to the left)
        px, py = -uy, ux
        # Half-width offset
        ox, oy = px * half_w, py * half_w

        # Rectangle corners in projected coords (A->B->C->D)
        A = (p0.x - ox, p0.y - oy)
        B = (p0.x + ox, p0.y + oy)
        C = (p1.x + ox, p1.y + oy)
        D = (p1.x - ox, p1.y - oy)

        # Convert back to lon/lat
        poly_lonlat = to_lonlat(inv, [A, B, C, D])

        rects.append(
            SlotRectangle(index=i, polygon_lonlat=poly_lonlat, start_m=start_d, end_m=end_d)
        )

    return rects


# ------------------------
# Streamlit App
# ------------------------

st.set_page_config(page_title="Path Slotting Map", layout="wide")
st.title("Path Slotting Map (Phase 1)")

st.markdown(
    """
This tool:
1. Lets you upload a CSV and **map columns** to: Lat, Long, Time, Operator, Signal, BTS ID.
2. **Sorts** by Time, plots samples on a map, and **computes total distance** along the path.
3. Performs **slotting**: given a **Slot Length (meters)**, it draws **rectangles (length = SL, width = user-selected — default 5 m)** that tile the path.
    """
)

uploaded = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df_raw = pd.read_csv(uploaded, encoding_errors="ignore")

    st.subheader("Preview")
    st.dataframe(df_raw.head(20), use_container_width=True)

    st.subheader("Map your columns")
    cols = list(df_raw.columns)

    col1, col2, col3 = st.columns(3)
    with col1:
        lat_col = st.selectbox("Latitude column", cols, index=None, placeholder="Select latitude")
        time_col = st.selectbox("Time column", cols, index=None, placeholder="Select time")
    with col2:
        lon_col = st.selectbox("Longitude column", cols, index=None, placeholder="Select longitude")
        op_col = st.selectbox("Operator column", cols, index=None, placeholder="Select operator")
    with col3:
        sig_col = st.selectbox("Signal column", cols, index=None, placeholder="Select signal")
        bts_col = st.selectbox("BTS ID column", cols, index=None, placeholder="Select BTS ID")

    proceed = lat_col and lon_col and time_col and op_col and sig_col and bts_col

    if not proceed:
        st.info("Select all six mappings to proceed.")

    if proceed:
        # Build normalized dataframe
        df = pd.DataFrame({
            "lat": pd.to_numeric(df_raw[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df_raw[lon_col], errors="coerce"),
            "time": pd.to_datetime(df_raw[time_col], errors="coerce"),
            "operator": df_raw[op_col].astype(str),
            "signal": df_raw[sig_col].astype(str),
            "bts_id": df_raw[bts_col].astype(str),
        })

        # Extract numeric signal if possible (first token), e.g., "-110 dBm" -> -110
        df["signal_num"] = pd.to_numeric(
            df["signal"].astype(str).str.split().str[0],
            errors="coerce"
        )

        # Drop invalid rows
        df = df.dropna(subset=["lat", "lon"]).copy()

        # Parse time but don't require it; allow fallback ordering
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["__order__"] = np.arange(len(df))
        order_choice = st.radio("Order path by", ["Time", "Original row order"], index=0, horizontal=True)
        if order_choice == "Time" and df["time"].notna().sum() >= 2:
            df = df.sort_values("time")
        else:
            if order_choice == "Time":
                st.warning("Time column couldn't be parsed reliably; falling back to original row order.")
            df = df.sort_values("__order__")

        df = df.reset_index(drop=True)

        # Sort by time
        

        if len(df) < 2:
            st.warning("Need at least two valid samples after cleaning to proceed.")
            st.stop()

        # Compute total distance
        # Distance calculation (with optional jump filtering)
        st.subheader("Distance calculation")
        filt = st.checkbox("Filter unrealistic jumps when computing distance", value=True)
        max_step = st.number_input("Max step distance (m)", min_value=1.0, value=200.0, step=10.0)

        df_for_dist = df.copy()
        if filt:
            keep = [True]
            for i in range(1, len(df_for_dist)):
                step = haversine_m(
                    df_for_dist["lat"].iloc[i-1], df_for_dist["lon"].iloc[i-1],
                    df_for_dist["lat"].iloc[i], df_for_dist["lon"].iloc[i]
                )
                keep.append(step <= max_step)
            df_for_dist = df_for_dist[pd.Series(keep)].reset_index(drop=True)

        total_dist_m = total_path_distance_m(df_for_dist["lat"].values, df_for_dist["lon"].values)

        # Slot length input
        st.subheader("Slotting Parameters")
        sl = st.number_input("Slot Length (meters)", min_value=1.0, value=50.0, step=1.0)
        width_m = st.number_input("Slot Width (meters)", min_value=0.5, value=50.0, step=0.5)
        signal_thresh = st.number_input("Good signal threshold T (dBm)", value=-124.0, step=1.0)

        # Generate rectangles
        rects = generate_slot_rectangles(df["lat"].values, df["lon"].values, float(sl), width_m=width_m)
        n_rects = len(rects)
        if n_rects == 0:
            st.warning("No slot rectangles generated. Try increasing Slot Length or check your path.")

        # Build path coordinates for PyDeck PathLayer
        path_coords = df[["lon", "lat"]].values.tolist()  # [[lon, lat], ...]

        # Build polygon data list for PyDeck
        poly_data = []
        for r in rects:
            poly_data.append({
                "polygon": [[lon, lat] for (lon, lat) in r.polygon_lonlat],
                "slot": r.index,
                "start_m": round(r.start_m, 2),
                "end_m": round(r.end_m, 2),
            })

        # --- Slot coverage stats and colors ---
        slot_polys = [Polygon([(lon, lat) for (lon, lat) in r.polygon_lonlat]) for r in rects]
        pts = [Point(lon, lat) for lon, lat in df[["lon", "lat"]].to_numpy()]
        sig_vals = df["signal_num"].to_numpy() if "signal_num" in df.columns else np.full(len(df), np.nan)

        def pct_to_color(p):
            # red (0%) → green (100%), grey if NaN (no samples)
            if p is None or (isinstance(p, float) and np.isnan(p)):
                return [180, 180, 180, 120]
            p01 = max(0.0, min(1.0, p/100.0))
            r = int(round(255 * (1.0 - p01)))
            g = int(round(255 * p01))
            return [r, g, 0, 140]

        slot_rows = []
        for idx, (r, poly) in enumerate(zip(rects, slot_polys)):
            inside_idx = [i for i, pt in enumerate(pts) if poly.intersects(pt)]
            total = len(inside_idx)
            good = int(np.sum(~np.isnan(sig_vals[inside_idx]) & (sig_vals[inside_idx] >= signal_thresh))) if total > 0 else 0
            bad = total - good
            pct = (good / total * 100.0) if total > 0 else np.nan
            color = pct_to_color(pct)

            # attach to map data
            poly_data[idx]["n_samples"] = total
            poly_data[idx]["n_good"] = good
            poly_data[idx]["n_bad"] = bad
            poly_data[idx]["pct"] = None if (isinstance(pct, float) and np.isnan(pct)) else round(pct, 2)
            poly_data[idx]["color"] = color

            slot_rows.append({
                "slot": r.index,
                "start_m": round(r.start_m, 2),
                "end_m": round(r.end_m, 2),
                "n_samples": total,
                "n_good": good,
                "n_bad": bad,
                "coverage_pct": None if (isinstance(pct, float) and np.isnan(pct)) else round(pct, 2),
            })

        stats_df = pd.DataFrame(slot_rows).sort_values("slot").reset_index(drop=True)

        # Map view state
        center_lat = float(df["lat"].mean())
        center_lon = float(df["lon"].mean())
        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=15, pitch=0)

        # Layers
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position="[lon, lat]",
            get_radius=7,
            radius_min_pixels=4,
            radius_max_pixels=14,
            get_fill_color=[0, 122, 255, 200],  # blue samples
            get_line_color=[0, 0, 0, 200],
            pickable=True,
        )

        polygon_layer = pdk.Layer(
            "PolygonLayer",
            data=poly_data,
            get_polygon="polygon",
            stroked=True,
            filled=True,
            get_line_width=4,
            get_line_color=[20, 20, 20, 220],  # dark outline
            get_fill_color="color",             # gradient by coverage
            pickable=True,
        )

        deck = pdk.Deck(
            layers=[polygon_layer, scatter_layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>Slot:</b> {slot}<br/><b>Coverage:</b> {pct}%<br/><b>Good/Total:</b> {n_good}/{n_samples}<br/><b>Start–End (m):</b> {start_m}–{end_m}",
                "style": {"fontSize": "12px"},
            },
            map_provider="carto",
            map_style="light",
        )

        st.subheader("Map")
        st.pydeck_chart(deck, use_container_width=True)

        st.subheader("Slot Coverage Table")
        st.dataframe(stats_df, use_container_width=True)
        st.download_button(
            "Download coverage CSV",
            data=stats_df.to_csv(index=False),
            file_name="slot_coverage.csv",
            mime="text/csv",
        )

        st.caption("Legend: blue dots = samples; red→green rectangles = slots by coverage")

        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Total distance", f"{total_dist_m/1000:.3f} km")
        with colB:
            st.metric("Samples", f"{len(df)}")
        with colC:
            st.metric("Slots (rectangles)", f"{n_rects}")

        # Optional: Full Coverage Map (slot-level good/bad/skip view)
        full_cov = st.checkbox("Full Coverage Map", value=False)
        if full_cov:
            # Build initial labels per slot
            label_map = {}
            for _, row in stats_df.iterrows():
                if row["n_samples"] == 0:
                    label_map[row["slot"]] = "Skip"
                elif row["n_good"] >= 1:
                    label_map[row["slot"]] = "Good"
                else:
                    label_map[row["slot"]] = "Bad"

            # Convert runs of consecutive Skip (length > 1) to Bad
            slots_sorted = stats_df["slot"].tolist()
            i = 0
            while i < len(slots_sorted):
                s = slots_sorted[i]
                if label_map.get(s) == "Skip":
                    j = i + 1
                    while j < len(slots_sorted) and label_map.get(slots_sorted[j]) == "Skip":
                        j += 1
                    if j - i > 1:
                        for k in range(i, j):
                            label_map[slots_sorted[k]] = "Bad"
                    i = j
                else:
                    i += 1

            # Build polygons for the Full Coverage view
            def label_color(lbl):
                if lbl == "Good":
                    return [0, 180, 0, 160]
                if lbl == "Bad":
                    return [210, 0, 0, 160]
                return [150, 150, 150, 120]  # Skip

            poly_data_full = []
            for r in rects:
                lbl = label_map.get(r.index, "Skip")
                poly_data_full.append({
                    "polygon": [[lon, lat] for (lon, lat) in r.polygon_lonlat],
                    "slot": r.index,
                    "label": lbl,
                    "color": label_color(lbl),
                    "start_m": round(r.start_m, 2),
                    "end_m": round(r.end_m, 2),
                })

            polygon_layer_full = pdk.Layer(
                "PolygonLayer",
                data=poly_data_full,
                get_polygon="polygon",
                stroked=True,
                filled=True,
                get_line_width=3,
                get_line_color=[30, 30, 30, 220],
                get_fill_color="color",
                pickable=True,
            )

            deck_full = pdk.Deck(
                layers=[polygon_layer_full],
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>Slot:</b> {slot}<br/><b>Label:</b> {label}<br/><b>Start–End (m):</b> {start_m}–{end_m}",
                    "style": {"fontSize": "12px"},
                },
                map_provider="carto",
                map_style="light",
            )

            st.subheader("Full Coverage Map")
            st.pydeck_chart(deck_full, use_container_width=True)

            # Coverage metrics
            good_slots = sum(1 for v in label_map.values() if v == "Good")
            skip_slots = sum(1 for v in label_map.values() if v == "Skip")
            total_slots = len(rects)
            denom = total_slots - skip_slots
            coverage_pct = (good_slots / denom * 100.0) if denom > 0 else float("nan")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Good slots", good_slots)
            with m2:
                st.metric("Bad slots", total_slots - good_slots - skip_slots)
            with m3:
                st.metric("Skipped slots", skip_slots)
            with m4:
                st.metric("Total coverage", f"{coverage_pct:.2f} %")

        with st.expander("Download slot polygons (GeoJSON-like)"):
            # Simple GeoJSON FeatureCollection-like dict (not strict spec for holes/properties)
            features = []
            for r in rects:
                features.append({
                    "type": "Feature",
                    "properties": {
                        "slot": r.index,
                        "start_m": r.start_m,
                        "end_m": r.end_m,
                        "slot_length_m": float(sl),
                        "width_m": width_m,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[lon, lat] for (lon, lat) in r.polygon_lonlat + [r.polygon_lonlat[0]]]],
                    },
                })
            fc = {"type": "FeatureCollection", "features": features}
            json_str = json.dumps(fc, indent=2)
            st.download_button("Download GeoJSON", data=json_str, file_name="slots.geojson", mime="application/geo+json")

else:
    st.info("Upload a CSV to begin.")

