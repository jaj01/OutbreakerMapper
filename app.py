# app.py (updated)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import networkx as nx
from geopy.distance import geodesic
from pyvis.network import Network
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium
import numpy as np
import os
import json

# OPTIONAL: import the inference helper if you want to run counterfactuals in-app
# from inference import run_counterfactual_prediction

st.set_page_config(page_title="OutbreakMapper", layout="wide")

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("outbreaker_with_final_coords_cleaned.csv")
    df['state'] = df['state'].astype(str).str.strip().str.title()
    df['district'] = df['district'].astype(str).str.strip().str.title()
    if "year_week" not in df.columns:
        df["year_week"] = df["year"].astype(str) + "_W" + df["week"].astype(str)
    # unify lat/lon cols if present
    if "latitude_y" in df.columns and "longitude_y" in df.columns:
        df = df.rename(columns={"latitude_y": "latitude", "longitude_y": "longitude"})
    return df

df = load_data()

# --- Load nodes/centroids if available ---
@st.cache_data
def load_nodes(nodes_path="data/processed/nodes.csv"):
    if os.path.exists(nodes_path):
        ndf = pd.read_csv(nodes_path)
        ndf['state'] = ndf['state'].astype(str).str.strip().str.title()
        ndf['district'] = ndf['district'].astype(str).str.strip().str.title()
        return ndf
    # fallback try to build from df
    fallback = df.groupby(['state','district'], as_index=False).agg({
        'latitude': 'mean', 'longitude': 'mean'
    })
    fallback['node_id'] = range(len(fallback))
    return fallback

nodes_df = load_nodes()

# --- Load predictions (if exist) ---
@st.cache_data
def load_predictions(path="predictions.csv"):
    if os.path.exists(path):
        preds = pd.read_csv(path)
        # Expect columns: time_batch_idx, node_global_idx, state, district, y_true, y_pred, year_week (optional)
        if 'year_week' not in preds.columns:
            # try infer or set 'all_weeks'
            if 'time' in preds.columns:
                preds['year_week'] = pd.to_datetime(preds['time']).dt.to_period('W').astype(str)
            else:
                preds['year_week'] = preds.get('year_week', 'all_weeks')
        preds['state'] = preds['state'].astype(str).str.strip().str.title()
        preds['district'] = preds['district'].astype(str).str.strip().str.title()
        return preds
    else:
        return pd.DataFrame()  # empty DataFrame fallback

preds_df = load_predictions()

# --- Load simplified geojson if you have one ---
@st.cache_data
def load_geojson(path="artifacts/india_districts_simplified.geojson"):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

geojson = load_geojson()

# --- Sidebar filters ---
st.sidebar.header("⚙️ Filters & Precaution thresholds")

# Disease filter
diseases = ["All"] + sorted(df["disease_grouped"].dropna().unique().tolist())
selected_disease = st.sidebar.selectbox("Select Disease", diseases)

# State filter
states = ["All"] + sorted(df["state"].dropna().unique().tolist())
selected_states = st.sidebar.multiselect("Select States", states, default=["All"])

# Hotspot thresholds (tunable)
st.sidebar.markdown("### Hotspot thresholds (tuneable)")
HIGH_CASES_THRESH = st.sidebar.number_input("High raw-case threshold (absolute)", value=50, step=10)
PCT_INCREASE_THRESH = st.sidebar.number_input("Pct increase threshold (e.g., 30% -> 0.30)", value=0.30, step=0.05, format="%.2f")
INCIDENCE_RATE_THRESH = st.sidebar.number_input("Incidence threshold (pred/pop) e.g. 0.001", value=0.001, step=0.0005, format="%.4f")

# --- Apply filters ---
filtered_df = df.copy()
if selected_disease != "All":
    filtered_df = filtered_df[filtered_df["disease_grouped"] == selected_disease]

if "All" not in selected_states:
    filtered_df = filtered_df[filtered_df["state"].isin(selected_states)]

# --- Base Graph (neighbors by distance) ---
@st.cache_data
def build_base_graph(df, dist_threshold=100):
    # derive centroids (unique district rows)
    districts = df.groupby(["state","district","latitude","longitude"]).size().reset_index().drop(columns=0)
    G_base = nx.Graph()
    for _, row in districts.iterrows():
        # if lat/lon missing, skip node (or add with None)
        G_base.add_node(row["district"], state=row["state"], pos=(row["longitude"], row["latitude"]))
    # compute pairwise distances (can be optimized)
    for i, row1 in districts.iterrows():
        for j, row2 in districts.iterrows():
            if i < j:
                try:
                    if pd.isna(row1["latitude"]) or pd.isna(row2["latitude"]):
                        continue
                    dist = geodesic((row1["latitude"], row1["longitude"]),
                                    (row2["latitude"], row2["longitude"])).km
                    if dist <= dist_threshold:
                        G_base.add_edge(row1["district"], row2["district"], distance=dist)
                except Exception:
                    continue
    return G_base

G_base = build_base_graph(df)

# ================
# TABS (3 total)
# ================
tab1, tab2, tab3 = st.tabs(["🌐 Network Graph", "🗺️ Maps & Trends", "📈 Predictions"])

# ======================
# TAB 1: NETWORK GRAPH
# ======================
with tab1:
    st.header("🌐 Temporal Disease Spread Network")

    weeks = sorted(filtered_df["year_week"].dropna().unique())
    if weeks:
        week_idx = st.slider("Select Week", 0, len(weeks)-1, 0, key="week_slider_net")
        selected_week = weeks[week_idx]
        st.subheader(f"📅 Network for: **{selected_week}** | Disease: **{selected_disease}**")

        week_cases = filtered_df[filtered_df["year_week"] == selected_week].groupby("district")["cases"].sum().to_dict()

        G = G_base.copy()
        for n in G.nodes:
            G.nodes[n]["cases"] = week_cases.get(n, 0)
        for u, v, data in G.edges(data=True):
            cases_u = G.nodes[u]["cases"]
            cases_v = G.nodes[v]["cases"]
            data["weight"] = 1 / (1 + abs(cases_u - cases_v))

        net = Network(height="700px", width="100%", bgcolor="#222222", font_color="white")
        for n, data in G.nodes(data=True):
            net.add_node(
                n,
                label=f"{n} ({data['state']})",
                title=f"District: {n}<br>State: {data['state']}<br>Cases: {data['cases']}",
                size=max(5, data['cases']/10) if isinstance(data.get('cases'), (int,float)) else 5,
                color="red" if data.get('cases', 0) > 100 else "orange"
            )
        for u, v, data in G.edges(data=True):
            net.add_edge(u, v, value=data.get('weight', 1.0))

        html_str = net.generate_html()
        components.html(html_str, height=750, scrolling=True)
    else:
        st.warning("No weeks available in dataset.")

# ======================
# TAB 2: MAPS & TRENDS
# ======================
with tab2:
    st.header("🗺️ District Bubble Maps & Trends")

    # --- Bubble Map ---
    st.subheader(f"District Bubble Map ({selected_disease})")
    district_cases = filtered_df.groupby(["state", "district", "latitude", "longitude"])["cases"].sum().reset_index()

    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")

    for _, row in district_cases.iterrows():
        if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=max(3, float(row["cases"]) / 20),
                popup=f"{row['district']}, {row['state']}<br>Cases: {int(row['cases'])}",
                color="red",
                fill=True,
                fill_opacity=0.6
            ).add_to(m)

    st_folium(m, width=900, height=600)

    # --- Trends ---
    st.subheader(f"Top District Trends ({selected_disease})")
    top_districts = filtered_df.groupby("district")["cases"].sum().nlargest(5).index.tolist()
    trend_data = filtered_df[filtered_df["district"].isin(top_districts)]

    if not trend_data.empty:
        # Ensure year_week is sorted sequentially (string->period)
        try:
            trend_data['year_week_pd'] = pd.to_datetime(trend_data['reporting_date'])
        except:
            trend_data['year_week_pd'] = trend_data['year_week']
        chart = (
            alt.Chart(trend_data)
            .mark_line(point=True)
            .encode(
                x="year_week:T" if 'reporting_date' in trend_data.columns else "year_week:N",
                y="cases:Q",
                color="district:N",
                tooltip=["district", "year_week", "cases"]
            )
            .properties(width=800, height=400)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No trend data available for this disease.")

# ======================
# TAB 3: PREDICTIONS
# ======================
with tab3:
    st.header("📈 Model Predictions & Precautions")

    if preds_df.empty:
        st.warning("Predictions file not found at outputs/predictions.csv. Place predictions.csv at that path and reload.")
    else:
        # Week selection
        weeks_p = sorted(preds_df['year_week'].unique())
        if weeks_p:
            sel_week_idx = st.slider(
                "Select week (predictions)",
                0,
                len(weeks_p)-1,
                len(weeks_p)-1,
                key="week_slider_pred"
            )
            sel_week = weeks_p[sel_week_idx]
            st.subheader(f"Predictions for week: {sel_week}")
        else:
            st.warning("⚠️ No prediction weeks available. Please check predictions.csv.")
            sel_week = None
        #sel_week_idx = st.slider("Select week (predictions)", 0, max(0, len(weeks_p)-1), max(0, len(weeks_p)-1), key="week_slider_pred")
        #sel_week = weeks_p[sel_week_idx]
        #st.subheader(f"Predictions for: {sel_week}")


        # filter preds
        df_week = preds_df[preds_df['year_week'] == sel_week].copy()

        # merge centroids/pop if available
        if not nodes_df.empty:
            if 'node_id' in nodes_df.columns and 'node_global_idx' in df_week.columns:
                df_week = df_week.merge(nodes_df, left_on='node_global_idx', right_on='node_id', how='left')
            else:
                df_week = df_week.merge(nodes_df, on=['state', 'district'], how='left')

        # Top-K
        k = st.slider("Top K hotspots", 5, 50, 10, key="hotspot_k")
        df_week_sorted = df_week.sort_values('y_pred', ascending=False).reset_index(drop=True)
        topk_df = df_week_sorted.head(k).copy()
        topk_df['rank_pred'] = range(1, len(topk_df)+1)

        st.subheader(f"Top {k} predicted hotspots")
        st.dataframe(topk_df[['rank_pred','state','district','y_true','y_pred']])

        # Map view (prefer geojson->choropleth, otherwise folium bubble)
        st.subheader("Spatial view (predicted cases)")

        if geojson is not None:
            # Try to match district property in geojson
            sample_props = geojson['features'][0]['properties']
            geo_prop = None
            for c in ['DIST_NAME','DIST_NAME','DISTRICT','DIST_NAME','district','DIST']:
                if c in sample_props:
                    geo_prop = c
                    break

            if geo_prop:
                # Map predicted value into geojson properties (best-effort matching by uppercase name)
                mapping = {str(r['district']).upper(): float(r['y_pred']) for _, r in df_week.iterrows() if pd.notna(r['district'])}
                for feat in geojson['features']:
                    pname = str(feat['properties'].get(geo_prop, '')).upper()
                    feat['properties']['y_pred'] = mapping.get(pname, 0.0)

                m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")
                folium.Choropleth(
                    geo_data=geojson,
                    name='Predicted cases',
                    data=df_week,
                    columns=['district','y_pred'] if 'district' in df_week.columns else ['node_global_idx','y_pred'],
                    key_on=f'feature.properties.{geo_prop}',
                    fill_color='YlOrRd',
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name='Predicted cases'
                ).add_to(m)

                # add top-k markers
                for _, r in topk_df.iterrows():
                    lat = r.get('latitude') or r.get('Latitude')
                    lon = r.get('longitude') or r.get('Longitude')
                    if pd.notna(lat) and pd.notna(lon):
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=max(6, np.log1p(r['y_pred']) * 2),
                            popup=f"{r['state']} - {r['district']}<br>Pred: {r['y_pred']:.1f}",
                            color='crimson', fill=True, fill_opacity=0.9
                        ).add_to(m)
                st_folium(m, width=900, height=650)
            else:
                st.info("GeoJSON loaded but no matching district property found — falling back to bubble map.")
                geo_match_failed = True
        else:
            geo_match_failed = True

        if geojson is None or ('geo_match_failed' in locals() and geo_match_failed):
            # fallback bubble map using lat/lon
            if 'latitude' in df_week.columns and 'longitude' in df_week.columns:
                m2 = folium.Map(location=[20.5937,78.9629], zoom_start=5, tiles="CartoDB positron")
                for _, r in df_week.iterrows():
                    if pd.notna(r.get('latitude')) and pd.notna(r.get('longitude')):
                        folium.CircleMarker(
                            location=[r['latitude'], r['longitude']],
                            radius=max(3, float(r['y_pred']) / 10),
                            popup=f"{r['district']}, {r['state']}<br>Pred: {r['y_pred']:.1f}",
                            color='red', fill=True, fill_opacity=0.7
                        ).add_to(m2)
                st_folium(m2, width=900, height=650)
            else:
                st.warning("No coordinates available for bubble map. Provide nodes.csv with lat/lon or a geojson for choropleth.")

        # Precaution generation: display per-topk
        st.subheader("Precaution recommendations (top hotspots)")
        for _, r in topk_df.iterrows():
            # simple rule engine (tweakable)
            recs = []
            y_pred = float(r.get('y_pred', 0.0))
            y_prev = float(r.get('y_true', 0.0))
            pop = r.get('population') or r.get('Population') or None
            inc_rate = (y_pred / pop) if pop and pop > 0 else None

            if y_pred >= HIGH_CASES_THRESH:
                recs.append("Scale-up testing, contact tracing and isolation facilities.")
            if y_prev > 0 and (y_pred - y_prev) / max(1, y_prev) >= PCT_INCREASE_THRESH:
                recs.append("Increase surveillance and consider temporary mobility restrictions.")
            if inc_rate is not None and inc_rate > INCIDENCE_RATE_THRESH:
                recs.append("Prioritize targeted vaccination and public awareness in the district.")
            if y_pred >= 100:
                recs.append("Prepare hospital surge capacity and essential supplies (O2, beds).")
            if not recs:
                recs.append("Maintain general NPIs: mask use, testing, and hygiene promotion.")

            st.markdown(f"**{r['state']} — {r['district']}** (Pred: {y_pred:.1f}, Prev: {y_prev:.1f})")
            for rec in recs:
                st.write(f"- {rec}")
            st.write("---")

        # Download weekly predictions
        st.markdown("### Download")
        st.download_button(
            label="Download weekly predictions (CSV)",
            data=df_week.to_csv(index=False),
            file_name=f"predictions_{sel_week}.csv",
            mime="text/csv"
        )
# End of app
