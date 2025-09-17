import streamlit as st
import pandas as pd
import networkx as nx
from geopy.distance import geodesic
from pyvis.network import Network
import streamlit.components.v1 as components

# --- Load dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("outbreaker_with_final_coords_cleaned.csv")
    df['state'] = df['state'].astype(str).str.strip().str.title()
    df['district'] = df['district'].astype(str).str.strip().str.title()
    if "year_week" not in df.columns:
        df["year_week"] = df["year"].astype(str) + "_W" + df["week"].astype(str)
    return df

df = load_data()

# --- Build base graph (district adjacency) ---
@st.cache_data
def build_base_graph(df, dist_threshold=100):
    districts = df.groupby(["state","district","latitude","longitude"]).size().reset_index().drop(columns=0)
    G_base = nx.Graph()
    for _, row in districts.iterrows():
        G_base.add_node(row["district"], state=row["state"], pos=(row["longitude"], row["latitude"]))
    for i, row1 in districts.iterrows():
        for j, row2 in districts.iterrows():
            if i < j:
                dist = geodesic((row1["latitude"], row1["longitude"]), (row2["latitude"], row2["longitude"])).km
                if dist <= dist_threshold:
                    G_base.add_edge(row1["district"], row2["district"], distance=dist)
    return G_base

G_base = build_base_graph(df)

# --- Streamlit UI ---
st.title("🦠 OutbreakMapper: Temporal Disease Spread Network")
st.markdown("Explore disease spread across districts using a temporal network graph.")

# Slider for week selection
weeks = sorted(df["year_week"].unique())
week = st.slider("Select Week", 0, len(weeks)-1, 0)
selected_week = weeks[week]
st.subheader(f"📅 Showing network for: {selected_week}")

# --- Build network for selected week ---
week_cases = df[df["year_week"] == selected_week].groupby("district")["cases"].sum().to_dict()
G = G_base.copy()

# Update node attributes
for n in G.nodes:
    G.nodes[n]["cases"] = week_cases.get(n, 0)

# Update edge weights
for u, v, data in G.edges(data=True):
    cases_u = G.nodes[u]["cases"]
    cases_v = G.nodes[v]["cases"]
    data["weight"] = 1 / (1 + abs(cases_u - cases_v))

# --- Create PyVis network ---
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

net.save_graph("network_temp.html")

# --- Embed into Streamlit ---
HtmlFile = open("network_temp.html", "r", encoding="utf-8")
components.html(HtmlFile.read(), height=750)
