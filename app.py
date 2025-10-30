# app.py - Robust OutbreakMapper Streamlit app (patched)
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import altair as alt
import networkx as nx
from geopy.distance import geodesic
from pyvis.network import Network
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium

# Try to import the inference helpers if present (optional)
try:
    from inference import load_model, run_counterfactual_prediction
    HAS_INFERENCE = True
except Exception:
    HAS_INFERENCE = False

# Page config
st.set_page_config(page_title="OutbreakMapper", layout="wide")

# --------------------
# CACHED LOADERS
# --------------------
@st.cache_data
def load_outbreak_data(path="outbreaker_with_final_coords_cleaned.csv"):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # ensure columns exist before manipulating
    if 'state' in df.columns:
        df['state'] = df['state'].astype(str).str.strip().str.title()
    else:
        df['state'] = ""
    if 'district' in df.columns:
        df['district'] = df['district'].astype(str).str.strip().str.title()
    else:
        df['district'] = ""
    # ensure year_week
    if 'year_week' not in df.columns and 'year' in df.columns and 'week' in df.columns:
        df['year_week'] = df['year'].astype(str) + "_W" + df['week'].astype(str)
    # unify lat/lon columns
    if 'latitude_y' in df.columns and 'longitude_y' in df.columns:
        df = df.rename(columns={'latitude_y': 'latitude', 'longitude_y': 'longitude'})
    return df

@st.cache_data
def load_nodes(nodes_path="data/processed/nodes.csv"):
    if os.path.exists(nodes_path):
        ndf = pd.read_csv(nodes_path)
        ndf.columns = [c.strip() for c in ndf.columns]
        if 'state' in ndf.columns:
            ndf['state'] = ndf['state'].astype(str).str.title().str.strip()
        if 'district' in ndf.columns:
            ndf['district'] = ndf['district'].astype(str).str.title().str.strip()
        return ndf
    # fallback: empty DF (app will derive centroids from outbreak if needed)
    return pd.DataFrame()

@st.cache_data
def load_predictions(path="predictions.csv.csv"):
    if os.path.exists(path):
        p = pd.read_csv(path)
        p.columns = [c.strip() for c in p.columns]
        if 'year_week' not in p.columns and 'time' in p.columns:
            try:
                p['year_week'] = pd.to_datetime(p['time']).dt.to_period('W').astype(str)
            except Exception:
                p['year_week'] = p.get('year_week', 'all_weeks')
        elif 'year_week' not in p.columns:
            p['year_week'] = p.get('year_week', 'all_weeks')
        # normalize names
        if 'state' in p.columns:
            p['state'] = p['state'].astype(str).str.strip().str.title()
        if 'district' in p.columns:
            p['district'] = p['district'].astype(str).str.strip().str.title()
        # Ensure numeric y_pred and y_true exist for safety
        if 'y_pred' in p.columns:
            p['y_pred'] = pd.to_numeric(p['y_pred'], errors='coerce').fillna(0.0)
        if 'y_true' in p.columns:
            p['y_true'] = pd.to_numeric(p['y_true'], errors='coerce').fillna(0.0)
        return p
    return pd.DataFrame()

@st.cache_data
def load_geojson(path="artifacts/india_districts_simplified.geojson"):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# --------------------
# Data
# --------------------
df = load_outbreak_data()
nodes_df = load_nodes()
preds_df = load_predictions()
geojson = load_geojson()

# --------------------
# Sidebar: filters & simulation
# --------------------
st.sidebar.header("Filters & Simulation")
diseases = ["All"] + (sorted(df['disease_grouped'].dropna().unique().tolist()) if not df.empty and 'disease_grouped' in df.columns else [])
selected_disease = st.sidebar.selectbox("Select Disease", diseases, index=0 if len(diseases)>0 else 0)

states = ["All"] + (sorted(df['state'].dropna().unique().tolist()) if not df.empty and 'state' in df.columns else [])
selected_states = st.sidebar.multiselect("Select States", states, default=["All"])

# NOTE: Hotspot thresholds removed from sidebar per request.
# Use fixed thresholds inside the precaution engine below.
# Constants (fixed)
FIXED_HIGH_CASES_THRESH = 50
FIXED_PCT_INCREASE_THRESH = 0.30
FIXED_INCIDENCE_RATE_THRESH = 0.001

# Counterfactual controls
st.sidebar.markdown("### Counterfactual (mobility)")
mob_scale = st.sidebar.slider("Mobility scale (0=lockdown, 1=no change)", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key="mob_scale")
run_cf = st.sidebar.button("Run counterfactual") if HAS_INFERENCE else st.sidebar.button("Run counterfactual (disabled)")

if not HAS_INFERENCE and run_cf:
    st.sidebar.warning("Inference module not found. Place inference.py (with load_model + run_counterfactual_prediction) in the app folder.")

# --------------------
# Apply filters
# --------------------
filtered_df = df.copy() if not df.empty else pd.DataFrame()
if selected_disease and selected_disease != "All" and 'disease_grouped' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['disease_grouped'] == selected_disease]
if selected_states and "All" not in selected_states and 'state' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['state'].isin(selected_states)]

# --------------------
# Helper: build base graph (spatial neighbors)
# --------------------
@st.cache_data
def build_base_graph(df_local, dist_threshold=120):
    if df_local.empty:
        return nx.Graph()
    # unique district centroids
    # handle possible column name inconsistencies
    lat_col = 'latitude' if 'latitude' in df_local.columns else None
    lon_col = 'longitude' if 'longitude' in df_local.columns else None
    districts = None
    if lat_col and lon_col:
        districts = df_local.groupby(['state','district',lat_col,lon_col]).size().reset_index().drop(columns=0)
        districts = districts.rename(columns={lat_col:'latitude', lon_col:'longitude'})
    else:
        # fallback: group only by state/district (no coords)
        districts = df_local.groupby(['state','district']).size().reset_index().drop(columns=0)
        districts['latitude'] = np.nan
        districts['longitude'] = np.nan

    G = nx.Graph()
    # add nodes
    for _, r in districts.iterrows():
        name = r['district']
        G.add_node(name, state=r['state'], latitude=r['latitude'], longitude=r['longitude'])
    # pairwise edges (simple O(N^2); ok for ~700 nodes)
    for i, r1 in districts.iterrows():
        for j, r2 in districts.iterrows():
            if i < j:
                lat1, lon1 = r1['latitude'], r1['longitude']
                lat2, lon2 = r2['latitude'], r2['longitude']
                if pd.isna(lat1) or pd.isna(lat2):
                    continue
                try:
                    d = geodesic((lat1, lon1), (lat2, lon2)).km
                except Exception:
                    continue
                if d <= dist_threshold:
                    G.add_edge(r1['district'], r2['district'], distance=d)
    return G

G_base = build_base_graph(df)

# --------------------
# Layout: tabs
# --------------------
tab1, tab2, tab3 = st.tabs(["🌐 Network Graph", "🗺️ Maps & Trends", "📈 Predictions & Simulation"])

# ======================
# TAB 1: NETWORK GRAPH
# ======================
with tab1:
    st.header("🌐 Temporal Disease Spread Network")
    weeks = sorted(filtered_df['year_week'].dropna().unique().tolist()) if not filtered_df.empty and 'year_week' in filtered_df.columns else []
    if not weeks:
        st.info("No weeks available in outbreak data (check 'year_week' column).")
    else:
        # ensure slider default is valid and unique key
        week_idx = st.slider("Select Week (network)", min_value=0, max_value=len(weeks)-1, value=len(weeks)-1, step=1, key="week_net")
        selected_week = weeks[week_idx]
        st.subheader(f"Network for {selected_week} | Disease: {selected_disease}")

        week_cases = filtered_df[filtered_df['year_week'] == selected_week].groupby('district')['cases'].sum().to_dict()
        G = G_base.copy()
        # set node cases
        for n in G.nodes:
            G.nodes[n]['cases'] = week_cases.get(n, 0)
        # set weights based on similarity
        for u, v, d in G.edges(data=True):
            cu = G.nodes[u].get('cases', 0)
            cv = G.nodes[v].get('cases', 0)
            d['weight'] = 1.0 / (1.0 + abs(cu - cv))

        # pyvis visual
        net = Network(height="700px", width="100%", bgcolor="#111111", font_color="white")
        for n, data in G.nodes(data=True):
            size_val = 5
            try:
                size_val = max(5, float(data.get('cases', 0)) / 10.0)
            except Exception:
                size_val = 5
            color = "red" if data.get('cases', 0) > 100 else "orange"
            net.add_node(n,
                         label=f"{n}",
                         title=f"{n} ({data.get('state','')})\nCases: {int(data.get('cases',0))}",
                         size=size_val, color=color)
        for u, v, d in G.edges(data=True):
            net.add_edge(u, v, value=d.get('weight', 1.0))
        html_str = net.generate_html()
        components.html(html_str, height=750, scrolling=True)

# ======================
# TAB 2: MAPS & TRENDS
# ======================
with tab2:
    st.header("🗺️ District Bubble Maps & Trends")
    st.subheader(f"District Bubble Map ({selected_disease})")

    if filtered_df.empty:
        st.info("No outbreak rows to show (after filters).")
    else:
        district_cases = filtered_df.groupby(['state','district','latitude','longitude'])['cases'].sum().reset_index()
        m = folium.Map(location=[20.5937,78.9629], zoom_start=5, tiles="CartoDB positron")
        for _, r in district_cases.iterrows():
            lat, lon = r['latitude'], r['longitude']
            if pd.notna(lat) and pd.notna(lon):
                folium.CircleMarker(location=[lat, lon],
                                    radius=max(3, float(r['cases'])/20),
                                    popup=f"{r['district']}, {r['state']}<br>Cases: {int(r['cases'])}",
                                    color="red", fill=True, fill_opacity=0.6).add_to(m)
        st_folium(m, width=900, height=600)

        # Trends for top districts
        st.subheader("Top District Trends")
        top5 = filtered_df.groupby('district')['cases'].sum().nlargest(5).index.tolist()
        trend_data = filtered_df[filtered_df['district'].isin(top5)].copy()
        if trend_data.empty:
            st.info("No trend data available.")
        else:
            # if reporting_date exists, use it else year_week
            x_col = 'reporting_date' if 'reporting_date' in trend_data.columns else 'year_week'
            # handle date vs categorical axis
            if x_col == 'reporting_date':
                trend_data['reporting_date'] = pd.to_datetime(trend_data['reporting_date'], errors='coerce')
                chart = (alt.Chart(trend_data)
                         .mark_line(point=True)
                         .encode(x="reporting_date:T", y="cases:Q", color="district:N", tooltip=["district", "year_week", "cases"])
                         .properties(width=800, height=400))
            else:
                chart = (alt.Chart(trend_data)
                         .mark_line(point=True)
                         .encode(x=alt.X("year_week:N", sort=None), y="cases:Q", color="district:N", tooltip=["district", "year_week", "cases"])
                         .properties(width=800, height=400))
            st.altair_chart(chart, use_container_width=True)

# ======================
# TAB 3: PREDICTIONS & SIMULATION
# ======================
with tab3:
    st.header("📈 Predictions & Counterfactual Simulation")

    if preds_df.empty:
        st.warning("Predictions CSV not found at 'outputs/predictions.csv'. Place predictions.csv there or update the loader path.")
    else:
        # safe weeks list
        weeks_p = sorted(preds_df['year_week'].dropna().astype(str).unique().tolist())
        if not weeks_p:
            st.warning("No valid 'year_week' values in predictions.csv.")
        else:
            # guard slider; use unique key
           # Guard slider creation if only one week available
            if len(weeks_p) > 1:
                sel_idx = st.slider(
                    "Select prediction week",
                    min_value=0,
                    max_value=len(weeks_p) - 1,
                    value=len(weeks_p) - 1,
                    step=1,
                    key="week_pred"
                )
                sel_week = weeks_p[sel_idx]
            else:
                sel_week = weeks_p[0]
                st.info(f"Only one prediction week available: **{sel_week}**")
                st.subheader(f"Predictions for {sel_week}")

            df_week = preds_df[preds_df['year_week'].astype(str) == sel_week].copy()
            if df_week.empty:
                st.info("No predictions for the selected week.")
            else:
                # merge centroids if nodes_df present
                if not nodes_df.empty:
                    if not isinstance(df_week, pd.DataFrame):
                        st.error("Internal error: df_week must be a DataFrame.")
                    else:
                        # Ensure 'y_true' column exists — if not create from 'y_true' in preds or default 0
                        if 'y_true' not in df_week.columns:
                            # Some prediction CSVs store ground truth under different names; try a few fallbacks
                            fallbacks = ['true', 'label', 'actual', 'y_true_values']
                            found = False
                            for f in fallbacks:
                                if f in df_week.columns:
                                    df_week['y_true'] = df_week[f]
                                    found = True
                                    break
                            if not found:
                                df_week['y_true'] = 0.0
                    
                        # Ensure 'y_pred' column exists
                        if 'y_pred' not in df_week.columns:
                            fallbacks = ['pred', 'prediction', 'y_pred_values', 'yhat']
                            found = False
                            for f in fallbacks:
                                if f in df_week.columns:
                                    df_week['y_pred'] = df_week[f]
                                    found = True
                                    break
                            if not found:
                                st.warning("Predictions file does not contain 'y_pred' column; creating a zero column.")
                                df_week['y_pred'] = 0.0
                    
                        # Coerce to numeric safely (operates on Series)
                        df_week['y_true'] = pd.to_numeric(df_week['y_true'], errors='coerce')
                        df_week['y_pred'] = pd.to_numeric(df_week['y_pred'], errors='coerce')
                    
                        # Fill NaNs with 0.0 (or you can choose another sentinel)
                        df_week['y_true'] = df_week['y_true'].fillna(0.0)
                        df_week['y_pred'] = df_week['y_pred'].fillna(0.0)
                    
                        # Normalize lat/lon column names (many variants exist)
                        lat_candidates = ['latitude', 'lat', 'Latitude', 'LAT']
                        lon_candidates = ['longitude', 'lon', 'Longitude', 'LON']
                        lat_col = next((c for c in lat_candidates if c in df_week.columns), None)
                        lon_col = next((c for c in lon_candidates if c in df_week.columns), None)
                    
                        if lat_col and lon_col:
                            df_week['latitude'] = pd.to_numeric(df_week[lat_col], errors='coerce')
                            df_week['longitude'] = pd.to_numeric(df_week[lon_col], errors='coerce')
                        else:
                            # Try merging nodes_df later will add coords; warn now
                            df_week['latitude'] = df_week.get('latitude', pd.Series([pd.NA]*len(df_week)))
                            df_week['longitude'] = df_week.get('longitude', pd.Series([pd.NA]*len(df_week)))
                            st.info("Latitude/Longitude not found in predictions; attempting to merge with nodes.csv if available.")
                    
                        # Optional: ensure state/district columns are normalized (useful for merges)
                        if 'state' in df_week.columns:
                            df_week['state'] = df_week['state'].astype(str).str.strip().str.title()
                        if 'district' in df_week.columns:
                            df_week['district'] = df_week['district'].astype(str).str.strip().str.title()
                    
                        # Final sanity: if y_pred all zeros warn the user (common when column mismatched)
                        if df_week['y_pred'].abs().sum() == 0.0:
                            st.warning("All predicted values for this week are zero. Please verify the 'y_pred' column in the predictions CSV.")
                    
                    # Now df_week has y_true, y_pred, latitude, longitude columns as numeric Series safe for later plotting/logic.
                    if 'node_id' in nodes_df.columns and 'node_global_idx' in df_week.columns:
                        df_week = df_week.merge(nodes_df, left_on='node_global_idx', right_on='node_id', how='left')
                    else:
                        # merge by state/district if possible
                        if set(['state','district']).issubset(nodes_df.columns) and set(['state','district']).issubset(df_week.columns):
                            df_week = df_week.merge(nodes_df, on=['state','district'], how='left')

                # Top-K hotspots
                k = st.slider("Top K hotspots", 5, 50, 10, key="hot_k")
                # ensure presence of 'y_pred' column
                if 'y_pred' not in df_week.columns and 'predicted_cases' in df_week.columns:
                    df_week['y_pred'] = pd.to_numeric(df_week['predicted_cases'], errors='coerce').fillna(0.0)
                #if 'y_true' not in df_week.columns:
                    #df_week['y_true'] = pd.to_numeric(df_week.get('y_true', 0.0), errors='coerce').fillna(0.0)

                sorted_week = df_week.sort_values('y_pred', ascending=False).reset_index(drop=True)
                topk = sorted_week.head(k).copy()
                topk['rank'] = range(1, len(topk)+1)
                st.subheader(f"Top {k} predicted hotspots")
                display_cols = ['rank','state','district','y_true','y_pred'] if set(['y_true','y_pred']).issubset(topk.columns) else topk.columns.tolist()
                st.dataframe(topk[display_cols])

                # Map: prefer geojson choropleth if available
                geo_failed = False
                if geojson is not None and len(geojson.get('features', []))>0:
                    # attempt to find district property name
                    sample_props = geojson['features'][0]['properties']
                    geo_prop = None
                    for cand in ['DIST_NAME','DISTRICT','district','DIST','NAME_2','DT_NAME','DIST_NAME']:
                        if cand in sample_props:
                            geo_prop = cand
                            break
                    if geo_prop:
                        # create mapping uppercase -> value
                        mapping = {}
                        for _, r in df_week.iterrows():
                            dname = str(r.get('district', '')).upper()
                            try:
                                mapping[dname] = float(r.get('y_pred', 0.0))
                            except Exception:
                                mapping[dname] = 0.0
                        # annotate features
                        for feat in geojson['features']:
                            pname = str(feat['properties'].get(geo_prop, '')).upper()
                            feat['properties']['y_pred'] = mapping.get(pname, 0.0)
                        m = folium.Map(location=[20.5937,78.9629], zoom_start=5, tiles="CartoDB positron")
                        try:
                            folium.Choropleth(geo_data=geojson, name='Predicted', data=df_week,
                                              columns=['district','y_pred'] if 'district' in df_week.columns else ['node_global_idx','y_pred'],
                                              key_on=f'feature.properties.{geo_prop}',
                                              fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2,
                                              legend_name='Predicted cases').add_to(m)
                        except Exception:
                            geo_failed = True
                        # add topk markers
                        for _, r in topk.iterrows():
                            lat = r.get('latitude') or r.get('Latitude') or None
                            lon = r.get('longitude') or r.get('Longitude') or None
                            if pd.notna(lat) and pd.notna(lon):
                                folium.CircleMarker(location=[lat, lon],
                                                    radius=max(6, np.log1p(float(r.get('y_pred',0.0)))*2),
                                                    popup=f"{r.get('district','')} ({r.get('state','')})\nPred: {float(r.get('y_pred',0.0)):.1f}",
                                                    color='crimson', fill=True, fill_opacity=0.9).add_to(m)
                        if not geo_failed:
                            st_folium(m, width=900, height=650)
                        else:
                            # fallback handled below
                            pass
                    else:
                        geo_failed = True
                else:
                    geo_failed = True

                if geo_failed:
                    # fallback bubble map using lat/lon in df_week
                    if 'latitude' in df_week.columns and 'longitude' in df_week.columns:
                        m2 = folium.Map(location=[20.5937,78.9629], zoom_start=5, tiles="CartoDB positron")
                        for _, r in df_week.iterrows():
                            lat, lon = r.get('latitude'), r.get('longitude')
                            if pd.notna(lat) and pd.notna(lon):
                                folium.CircleMarker(location=[lat, lon],
                                                    radius=max(3, float(r.get('y_pred',0.0))/10),
                                                    popup=f"{r.get('district','')}, {r.get('state','')}<br>Pred: {float(r.get('y_pred',0.0)):.1f}",
                                                    color='red', fill=True, fill_opacity=0.7).add_to(m2)
                        st_folium(m2, width=900, height=650)
                    else:
                        st.warning("No coordinates available for mapping. Provide nodes.csv with lat/lon or a geojson file.")

                # Precaution engine (multi-tier) — uses fixed thresholds declared above
                st.subheader("Precaution recommendations (Top hotspots)")
                def compute_severity_and_recs(row):
                    preds = float(row.get('y_pred', 0.0))
                    prev = float(row.get('y_true', 0.0))
                    pop = None
                    for key in ['population','Population','pop','POP']:
                        if key in row:
                            try:
                                pop = float(row.get(key))
                                break
                            except Exception:
                                pop = None
                    incidence = (preds / pop) if pop and pop>0 else 0.0
                    pct_inc = ((preds - prev)/prev) if prev>0 else (1.0 if preds>0 else 0.0)
                    # normalized scores using fixed thresholds
                    s_abs = preds / max(1.0, FIXED_HIGH_CASES_THRESH)
                    s_pct = min(pct_inc / max(1e-6, FIXED_PCT_INCREASE_THRESH), 5.0)
                    s_inc = incidence / max(1e-6, FIXED_INCIDENCE_RATE_THRESH)
                    score = 0.6*s_abs + 0.3*s_pct + 0.1*s_inc
                    # severity tier
                    if score < 1.5:
                        tier = "Low"
                        msgs = ["Maintain public advisories: testing availability, hygiene, mask promotion."]
                    elif score < 3.0:
                        tier = "Medium"
                        msgs = ["Increase surveillance and testing capacity; targeted public messaging; prepare local isolation facilities."]
                    elif score < 5.0:
                        tier = "High"
                        msgs = ["Scale up testing & contact tracing; consider targeted mobility restrictions; prioritize vaccination for vulnerable groups."]
                    else:
                        tier = "Critical"
                        msgs = ["Immediate surge preparedness (hospitals, O2); implement strict mobility reduction; mass testing & targeted lockdowns if necessary."]
                    return tier, score, msgs

                for _, r in topk.iterrows():
                    tier, score, msgs = compute_severity_and_recs(r)
                    ypred = float(r.get('y_pred', 0.0))
                    ytrue = float(r.get('y_true', 0.0))
                    st.markdown(f"**{r.get('state','')} — {r.get('district','')}**  Pred: **{ypred:.1f}** Prev: **{ytrue:.1f}**")
                    st.markdown(f"**Severity:** {tier}  |  **Score:** {score:.2f}")
                    for m in msgs:
                        st.write(f"- {m}")
                    st.write("---")

                # Download weekly predictions
                st.download_button("Download this week's predictions (CSV)", data=df_week.to_csv(index=False),
                                   file_name=f"predictions_{sel_week}.csv", mime="text/csv")

                # --------------- Counterfactual simulation (guarded) ---------------
                st.markdown("### Counterfactual simulation (mobility scaling)")
                # check prerequisites
                SNAPSHOT_DIR = "data/pyg_snapshots"
                MODEL_PATH = "artifacts/models/best_tgcn_gru.pth"

                def load_snapshot_sequence(target_week, window=4):
                    """Load up to 'window' snapshots ending at target_week from SNAPSHOT_DIR.
                       Expects files named '<year_week>.npz' containing arrays:
                        - node_features (N,F)
                        - edge_index (2,E)
                        - edge_weight (E)
                    """
                    if not os.path.exists(SNAPSHOT_DIR):
                        return None
                    all_files = sorted([f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".npz")])
                    weeks_avail = [os.path.splitext(f)[0] for f in all_files]
                    if target_week in weeks_avail:
                        idx = weeks_avail.index(target_week)
                        start = max(0, idx - window + 1)
                        sel_files = all_files[start: idx+1]
                    else:
                        # fallback: most recent window
                        sel_files = all_files[-window:]
                    if not sel_files:
                        return None
                    feats = []
                    eis = []
                    ews = []
                    for fname in sel_files:
                        try:
                            npz = np.load(os.path.join(SNAPSHOT_DIR, fname), allow_pickle=True)
                            feats.append(npz['node_features'])
                            eis.append(npz['edge_index'])
                            ews.append(npz['edge_weight'])
                        except Exception:
                            return None
                    return np.stack(feats, axis=0), eis, ews

                # Only allow running CF if inference module present
                if not HAS_INFERENCE:
                    st.info("Counterfactual simulation disabled — inference module not found. Add inference.py (with load_model+run_counterfactual_prediction) to enable.")
                else:
                    # Show info about availability
                    has_snap = os.path.exists(SNAPSHOT_DIR) and len([f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.npz')])>0
                    has_model = os.path.exists(MODEL_PATH)
                    if not has_snap:
                        st.warning(f"Snapshot folder '{SNAPSHOT_DIR}' missing or empty. Place .npz snapshot files to run CF.")
                    if not has_model:
                        st.warning(f"Model checkpoint not found at '{MODEL_PATH}'. Place trained checkpoint to run CF.")
                    if has_snap and has_model:
                        if st.button("Run counterfactual now (scale mobility)", key="run_cf_now"):
                            snap = load_snapshot_sequence(sel_week, window=4)
                            if snap is None:
                                st.error("Could not load required snapshot sequence for this week. Check file naming and contents.")
                            else:
                                node_feat_seq, edge_index_seq, edge_weight_seq = snap
                                try:
                                    # infer in_feats
                                    in_feats = node_feat_seq.shape[2]
                                    model = load_model(MODEL_PATH, in_feats=in_feats)
                                except Exception as e:
                                    st.error(f"Model load failed: {e}")
                                    model = None
                                if model is not None:
                                    st.info("Running baseline and counterfactual inferences (this may take a moment)...")
                                    preds_base = run_counterfactual_prediction(model, node_feat_seq, edge_index_seq, edge_weight_seq, mobility_scale=1.0)
                                    preds_cf = run_counterfactual_prediction(model, node_feat_seq, edge_index_seq, edge_weight_seq, mobility_scale=float(mob_scale))

                                    # prepare results dataframe using nodes_df if available
                                    mapping_nodes = nodes_df.copy() if not nodes_df.empty else None
                                    if mapping_nodes is not None and 'node_id' in mapping_nodes.columns:
                                        nodemap = mapping_nodes.sort_values('node_id').reset_index(drop=True)
                                    else:
                                        # fallback create node_id and mapping from df_week or nodes_df
                                        nodemap = mapping_nodes.reset_index().rename(columns={'index':'node_id'}) if mapping_nodes is not None else pd.DataFrame({'node_id': range(len(preds_base))})
                                    res = nodemap[['node_id','state','district','latitude','longitude']].copy() if not nodemap.empty else pd.DataFrame({'node_id': range(len(preds_base))})
                                    res['pred_base'] = preds_base[:len(res)]
                                    res['pred_cf'] = preds_cf[:len(res)]
                                    res['delta'] = res['pred_cf'] - res['pred_base']

                                    # show side-by-side small maps
                                    c1, c2, c3 = st.columns([1,1,0.8])
                                    with c1:
                                        st.markdown("**Baseline**")
                                        m_b = folium.Map(location=[20.5937,78.9629], zoom_start=5, tiles="CartoDB positron")
                                        for _, r in res.iterrows():
                                            lat, lon = r.get('latitude'), r.get('longitude')
                                            if pd.notna(lat) and pd.notna(lon):
                                                folium.CircleMarker(location=[lat, lon],
                                                                    radius=max(3, float(r['pred_base'])/10),
                                                                    popup=f"{r.get('district','')} ({r.get('state','')})\nPred: {r['pred_base']:.1f}",
                                                                    color='blue', fill=True, fill_opacity=0.7).add_to(m_b)
                                        st_folium(m_b, width=420, height=420)
                                    with c2:
                                        st.markdown(f"**Counterfactual (mob × {mob_scale:.2f})**")
                                        m_c = folium.Map(location=[20.5937,78.9629], zoom_start=5, tiles="CartoDB positron")
                                        for _, r in res.iterrows():
                                            lat, lon = r.get('latitude'), r.get('longitude')
                                            if pd.notna(lat) and pd.notna(lon):
                                                folium.CircleMarker(location=[lat, lon],
                                                                    radius=max(3, float(r['pred_cf'])/10),
                                                                    popup=f"{r.get('district','')} ({r.get('state','')})\nPred: {r['pred_cf']:.1f}",
                                                                    color='green', fill=True, fill_opacity=0.7).add_to(m_c)
                                        st_folium(m_c, width=420, height=420)
                                    with c3:
                                        st.markdown("**Delta (CF - Baseline)**")
                                        st.dataframe(res[['state','district','pred_base','pred_cf','delta']].sort_values('delta', ascending=False).head(15))

                                    # allow download
                                    st.download_button("Download counterfactual CSV",
                                                       data=res.to_csv(index=False),
                                                       file_name=f"counterfactual_{sel_week}_mob{mob_scale:.2f}.csv",
                                                       mime="text/csv")
