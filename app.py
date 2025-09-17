import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import altair as alt
import networkx as nx
from geopy.distance import geodesic
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="OutbreakMapper", layout="wide")

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("outbreaker_with_final_coords_cleaned.csv")
    df['state'] = df['state'].astype(str).str.strip().str.title()
    df['district'] = df['district'].astype(str).str.strip().str.title()
    if "year_week" not in df.columns:
        df["year_week"] = df["year"].astype(str) + "_W" + df["week"].astype(str)
    if "latitude_y" in df.columns and "longitude_y" in df.columns:
        df = df.rename(columns={"latitude_y": "latitude", "longitude_y": "longitude"})
    return df

df = load_data()

# --- Base Graph (neighbors by distance) ---
@st.cache_data
def build_base_graph(df, dist_threshold=100):
    districts = df.groupby(["state","district","latitude","longitude"]).size().reset_index().drop(columns=0)
    G_base = nx.Graph()
    for _, row in districts.iterrows():
        G_base.add_node(row["district"], state=row["state"], pos=(row["longitude"], row["latitude"]))
    for i, row1 in districts.iterrows():
        for j, row2 in districts.iterrows():
            if i < j:
                try:
                    dist = geodesic((row1["latitude"], row1["longitude"]),
                                    (row2["latitude"], row2["longitude"])).km
                    if dist <= dist_threshold:
                        G_base.add_edge(row1["district"], row2["district"], distance=dist)
                except:
                    continue
    return G_base

G_base = build_base_graph(df)

# --- Sidebar filters ---
st.sidebar.header("⚙️ Filters")
diseases = ["All"] + sorted(df["disease_grouped"].dropna().unique().tolist())
selected_disease = st.sidebar.selectbox("Select Disease", diseases)

# Filter dataset by disease
if selected_disease != "All":
    df = df[df["disease_grouped"] == selected_disease]

# --- Tabs ---
tab1, tab2 = st.tabs(["🌐 Network Graph", "🗺️ Heatmaps & Trends"])

# ======================
# TAB 1: NETWORK GRAPH
# ======================
with tab1:
    st.header("🌐 Temporal Disease Spread Network")

    weeks = sorted(df["year_week"].dropna().unique())
    if weeks:
        week_idx = st.slider("Select Week", 0, len(weeks)-1, 0, key="week_slider")
        selected_week = weeks[week_idx]
        st.subheader(f"📅 Network for: **{selected_week}** | Disease: **{selected_disease}**")

        # Cases for selected week
        week_cases = df[df["year_week"] == selected_week].groupby("district")["cases"].sum().to_dict()

        # Copy graph
        G = G_base.copy()
        for n in G.nodes:
            G.nodes[n]["cases"] = week_cases.get(n, 0)
        for u, v, data in G.edges(data=True):
            cases_u = G.nodes[u]["cases"]
            cases_v = G.nodes[v]["cases"]
            data["weight"] = 1 / (1 + abs(cases_u - cases_v))

        # Build PyVis
        net = Network(height="700px", width="100%", bgcolor="#222222", font_color="white")
        for n, data in G.nodes(data=True):
            net.add_node(
                n,
                label=f"{n} ({data['state']})",
                title=f"District: {n}<br>State: {data['state']}<br>Cases: {data['cases']}",
                size=max(5, data['cases']/10),
                color="red" if data['cases'] > 100 else "orange"
            )
        for u, v, data in G.edges(data=True):
            net.add_edge(u, v, value=data['weight'])

        html_str = net.generate_html()
        components.html(html_str, height=750, scrolling=True)
    else:
        st.warning("No weeks available in dataset.")

# ======================
# TAB 2: HEATMAPS & TRENDS
# ======================
with tab2:
    st.header("🗺️ State & District Analysis")

    # --- State-wise Heatmap ---
    st.subheader(f"State-wise Total Cases ({selected_disease})")
    state_cases = df.groupby("state")["cases"].sum().reset_index()

    try:
        india_states = gpd.read_file("india_maps/maps-master/States/IndiaState.shp")
        india_states["state"] = india_states["state"].astype(str).str.strip().str.title()
        state_cases["state"] = state_cases["state"].astype(str).str.strip().str.title()
        india_states = india_states.merge(state_cases, on="state", how="left").fillna(0)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        india_states.plot(column="cases", cmap="Reds", legend=True, ax=ax, edgecolor="black")
        ax.set_title(f"State-wise Outbreak Heatmap ({selected_disease})", fontsize=16)
        ax.axis("off")
        st.pyplot(fig)
    except Exception as e:
        st.warning("Could not load shapefile. Please upload India state boundaries.")

    # --- District-wise Trends ---
    st.subheader(f"Top Districts by Cases ({selected_disease})")
    top_districts = df.groupby("district")["cases"].sum().nlargest(5).index.tolist()
    trend_data = df[df["district"].isin(top_districts)]

    if not trend_data.empty:
        chart = (
            alt.Chart(trend_data)
            .mark_line(point=True)
            .encode(
                x="year_week:T",
                y="cases:Q",
                color="district:N",
                tooltip=["district", "year_week", "cases"]
            )
            .properties(width=800, height=400)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No trend data available for this disease.")
