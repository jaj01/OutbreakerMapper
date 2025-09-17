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
# --- Sidebar filters ---
st.sidebar.header("⚙️ Filters")

# Disease filter
diseases = ["All"] + sorted(df["disease_grouped"].dropna().unique().tolist())
selected_disease = st.sidebar.selectbox("Select Disease", diseases)

# State filter
states = ["All"] + sorted(df["state"].dropna().unique().tolist())
selected_states = st.sidebar.multiselect("Select States", states, default=["All"])

# Apply filters
filtered_df = df.copy()
if selected_disease != "All":
    filtered_df = filtered_df[filtered_df["disease_grouped"] == selected_disease]

if "All" not in selected_states:
    filtered_df = filtered_df[filtered_df["state"].isin(selected_states)]

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
# ======================
# TAB 1: NETWORK GRAPH
# ======================
with tab1:
    st.header("🌐 Temporal Disease Spread Network")

    weeks = sorted(filtered_df["year_week"].dropna().unique())
    if weeks:
        week_idx = st.slider("Select Week", 0, len(weeks)-1, 0, key="week_slider")
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
                radius=max(3, row["cases"]/20),
                popup=f"{row['district']}, {row['state']}<br>Cases: {row['cases']}",
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
