# app.py
import os
import io
import base64
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
from pyvis.network import Network
import networkx as nx

# Community detection (Louvain)
try:
    import community as community_louvain  # python-louvain
except Exception:
    community_louvain = None

# Box SDK
from boxsdk import OAuth2, Client
from boxsdk.exception import BoxAPIException

# -------------- CONFIG --------------
APP_TITLE = "Lab Literature Map"
DEFAULT_METADATA_FILENAME = "metadata.csv"       # stored in your Box folder
DEFAULT_TAXONOMY_FILENAME = "taxonomy.csv"       # optional: columns [child,parent]
PDF_MAX_MB = 50

# -------------- AUTH (shared code) --------------
def require_access_code():
    st.sidebar.header("Access")
    code = st.sidebar.text_input("Enter access code", type="password")
    saved = st.session_state.get("access_ok", False)
    if saved:
        return True
    if st.sidebar.button("Unlock"):
        if code and code == st.secrets.get("ACCESS_CODE", ""):
            st.session_state["access_ok"] = True
            return True
        else:
            st.error("Invalid access code.")
            return False
    return saved

# -------------- BOX HELPERS --------------
@st.cache_resource(show_spinner=False)
def _box_client():
    """
    Initialize Box client using one of:
      - Developer Token (quick start) via st.secrets["BOX_DEVELOPER_TOKEN"]
      - OAuth2 client credentials (JWT App User) via st.secrets["BOX_JWT_CONFIG_JSON"]
    """
    dev_token = st.secrets.get("BOX_DEVELOPER_TOKEN")
    jwt_json = st.secrets.get("BOX_JWT_CONFIG_JSON")

    if dev_token:
        oauth = OAuth2(
            client_id=None, client_secret=None, access_token=dev_token
        )
        client = Client(oauth)
        # Note: Developer tokens expire ~60 min. For production, use JWT App.
        return client, "dev_token"
    elif jwt_json:
        try:
            from boxsdk import JWTAuth
            config = json.loads(jwt_json)
            auth = JWTAuth.from_settings_dictionary(config)
            client = Client(auth)
            _ = client.user().get()
            return client, "jwt"
        except Exception as e:
            st.error(f"JWT auth failed: {e}")
            st.stop()
    else:
        st.error("No Box credentials found. Add BOX_DEVELOPER_TOKEN or BOX_JWT_CONFIG_JSON to secrets.")
        st.stop()

def _get_folder(client: Client, folder_id: str):
    try:
        return client.folder(folder_id).get()
    except BoxAPIException as e:
        st.error(f"Cannot access Box folder {folder_id}: {e}")
        st.stop()

def _list_folder_items(client: Client, folder_id: str) -> List[Dict]:
    items = []
    try:
        it = client.folder(folder_id=folder_id).get_items(limit=1000)
        for i in it:
            items.append({"id": i.id, "name": i.name, "type": i.type})
    except BoxAPIException as e:
        st.error(f"Error listing Box folder items: {e}")
    return items

def _get_file_by_name(client: Client, folder_id: str, filename: str):
    for it in _list_folder_items(client, folder_id):
        if it["type"] == "file" and it["name"] == filename:
            return client.file(it["id"]).get()
    return None

def _download_file_bytes(client: Client, file_id: str) -> bytes:
    try:
        stream = io.BytesIO()
        client.file(file_id).download_to(stream)
        return stream.getvalue()
    except BoxAPIException as e:
        st.error(f"Error downloading file: {e}")
        return b""

def _upload_file_bytes(client: Client, folder_id: str, filename: str, data: bytes, overwrite=True) -> Optional[str]:
    # Returns file_id
    try:
        existing = _get_file_by_name(client, folder_id, filename)
        if overwrite and existing:
            upd = client.file(existing.id).update_contents(io.BytesIO(data))
            return upd.id
        else:
            newf = client.folder(folder_id).upload_stream(io.BytesIO(data), filename)
            return newf.id
    except BoxAPIException as e:
        st.error(f"Upload failed: {e}")
        return None

def _ensure_shared_link(client: Client, file_id: str) -> str:
    try:
        f = client.file(file_id).get(fields=['shared_link'])
        if f.shared_link and "url" in f.shared_link:
            return f.shared_link["url"]
        # create a new open shared link
        f = client.file(file_id).update_info({"shared_link": {"access": "open"}})
        return f.shared_link["url"]
    except BoxAPIException as e:
        st.warning(f"Could not create shared link: {e}")
        return ""

# -------------- METADATA I/O --------------
def load_metadata(client: Client, folder_id: str, metadata_filename: str) -> pd.DataFrame:
    f = _get_file_by_name(client, folder_id, metadata_filename)
    if not f:
        # Create an empty metadata file
        df = pd.DataFrame(columns=["file_name","title","authors","year","keywords","notes","box_file_id","box_shared_url"])
        _upload_file_bytes(client, folder_id, metadata_filename, df.to_csv(index=False).encode("utf-8"))
        return df.copy()
    data = _download_file_bytes(client, f.id)
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception:
        df = pd.read_csv(io.BytesIO(data), encoding_errors="ignore")
    # Ensure required columns
    for col in ["file_name","title","authors","year","keywords","notes","box_file_id","box_shared_url"]:
        if col not in df.columns:
            df[col] = ""
    return df

def save_metadata(client: Client, folder_id: str, metadata_filename: str, df: pd.DataFrame):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    _upload_file_bytes(client, folder_id, metadata_filename, csv_bytes, overwrite=True)

def load_taxonomy(client: Client, folder_id: str, taxonomy_filename: str) -> pd.DataFrame:
    f = _get_file_by_name(client, folder_id, taxonomy_filename)
    if not f:
        # optional file; start empty
        return pd.DataFrame(columns=["child","parent"])
    data = _download_file_bytes(client, f.id)
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception:
        df = pd.read_csv(io.BytesIO(data), encoding_errors="ignore")
    if not {"child","parent"}.issubset(df.columns):
        st.warning("taxonomy.csv must have columns: child,parent")
        return pd.DataFrame(columns=["child","parent"])
    return df

# -------------- PDF PREVIEW --------------
def pdf_bytes_to_iframe(pdf_bytes: bytes, height=600) -> str:
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}" style="border:1px solid #ddd;border-radius:8px;"></iframe>'

# -------------- GRAPH / CLUSTERING --------------
def normalize_keywords(raw: str) -> List[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    # split by comma/semicolon/pipe
    parts = [p.strip() for p in re_split(raw)]
    # normalize lower case but keep a display copy elsewhere
    return sorted(set([p.lower() for p in parts if p]))

def re_split(s: str) -> List[str]:
    import re
    return re.split(r"[;,|]", s)

def build_keyword_graph(df: pd.DataFrame, taxonomy_df: pd.DataFrame) -> nx.Graph:
    """
    Build a graph that includes:
      - keyword nodes with edges for co-occurrence in the same document
      - taxonomy parent-child edges (directed stored as undirected with 'type' attr)
    """
    G = nx.Graph()
    # Add co-occurrence edges
    for _, row in df.iterrows():
        kws = normalize_keywords(row.get("keywords",""))
        for kw in kws:
            if kw not in G:
                G.add_node(kw, label=kw, kind="keyword")
        # co-occurrence pairs
        for i in range(len(kws)):
            for j in range(i+1, len(kws)):
                a, b = kws[i], kws[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1, type="co_occurrence")

    # Add taxonomy edges (broader→narrower)
    if not taxonomy_df.empty:
        for _, r in taxonomy_df.iterrows():
            child = str(r["child"]).strip().lower()
            parent = str(r["parent"]).strip().lower()
            if not child or not parent:
                continue
            for kw in (child, parent):
                if kw and kw not in G:
                    G.add_node(kw, label=kw, kind="keyword")
            if G.has_edge(child, parent):
                # preserve max weight but mark taxonomy
                G[child][parent]["type"] = "taxonomy"
                G[child][parent]["weight"] = max(G[child][parent].get("weight",1), 1)
            else:
                G.add_edge(child, parent, weight=1, type="taxonomy")
    return G

def assign_keyword_communities(G: nx.Graph) -> Dict[str, int]:
    if community_louvain is None or G.number_of_nodes() == 0:
        return {}
    # Use weights to inform communities
    part = community_louvain.best_partition(G, weight="weight", resolution=1.0, random_state=42)
    return part  # dict: node -> community id

def map_documents_to_clusters(df: pd.DataFrame, kw_partition: Dict[str,int]) -> Dict[int, List[int]]:
    """
    Assign each document to the community where it shares the most keywords.
    Returns: community_id -> list of df indices
    """
    cluster_docs = {}
    for idx, row in df.iterrows():
        kws = normalize_keywords(row.get("keywords",""))
        votes = {}
        for kw in kws:
            cid = kw_partition.get(kw)
            if cid is not None:
                votes[cid] = votes.get(cid, 0) + 1
        if votes:
            best = max(votes, key=votes.get)
        else:
            best = -1  # unclustered
        cluster_docs.setdefault(best, []).append(idx)
    return cluster_docs

def build_pyvis_graph(df: pd.DataFrame, G_kw: nx.Graph, kw_partition: Dict[str,int], filters: Dict) -> Network:
    """
    Construct a bipartite-ish visualization: keyword nodes + document nodes.
    - Keyword nodes colored by community
    - Document nodes connect to their keywords
    """
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="#222222", notebook=False, directed=False, cdn_resources="in_line")
    net.set_options(\"\"\"
    const options = {
      nodes: { shape: "dot", size: 12 },
      physics: { stabilization: true, solver: "forceAtlas2Based", timestep: 0.35 },
      interaction: { tooltipDelay: 120, hideEdgesOnDrag: false, multiselect: true, dragNodes: true },
      edges: { smooth: false, color: { inherit: true } }
    }
    \"\"\")

    # Color palette per community id
    def color_for(cid: Optional[int]) -> str:
        if cid is None:
            return "#888888"
        # generate stable pastel-ish colors
        rng = np.random.default_rng(cid + 13)
        r,g,b = rng.integers(60, 200, size=3)
        return f"rgb({r},{g},{b})"

    # Add keyword nodes
    for kw in G_kw.nodes():
        cid = kw_partition.get(kw)
        net.add_node(
            f"kw::{kw}",
            label=kw,
            title=f"Keyword: {kw}\\nCluster: {cid if cid is not None else 'unassigned'}",
            color=color_for(cid),
            physics=True
        )

    # Filter documents
    view_df = df.copy()
    if filters.get("year"):
        view_df = view_df[view_df["year"].astype(str).str.contains(filters["year"])]
    if filters.get("author"):
        view_df = view_df[view_df["authors"].astype(str).str.contains(filters["author"], case=False, na=False)]
    if filters.get("search_kw"):
        s = filters["search_kw"].strip().lower()
        view_df = view_df[view_df["keywords"].astype(str).str.lower().str.contains(s)]

    # Add document nodes and edges to keywords
    for _, row in view_df.iterrows():
        doc_id = f"doc::{row.get('file_name','')}"
        title = row.get("title","(untitled)")
        meta_t = f"Title: {title}\\nAuthors: {row.get('authors','')}\\nYear: {row.get('year','')}\\nKeywords: {row.get('keywords','')}"
        net.add_node(doc_id, label=title[:40], title=meta_t, shape="box", size=18, color="#1f77b4")

        kws = normalize_keywords(row.get("keywords",""))
        for kw in kws:
            if f"kw::{kw}" in [n["id"] for n in net.nodes]:
                net.add_edge(doc_id, f"kw::{kw}", value=2)

    # Strengthen keyword co-occurrence edges (optional)
    for a,b,data in G_kw.edges(data=True):
        w = data.get("weight", 1)
        edge_type = data.get("type","co_occurrence")
        dashes = True if edge_type == "taxonomy" else False
        net.add_edge(f"kw::{a}", f"kw::{b}", value=w, dashes=dashes)

    return net

# -------------- MAIN APP --------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("PDFs + keywords → interactive clusters. Synced to a shared Box folder.")

    if not require_access_code():
        st.stop()

    client, auth_mode = _box_client()

    # Sidebar setup
    st.sidebar.header("Data Source (Box)")
    folder_id = st.sidebar.text_input("Box Folder ID", value=st.secrets.get("BOX_FOLDER_ID",""), help="The ID from your Box folder URL (the number after /folder/)")
    metadata_filename = st.sidebar.text_input("Metadata filename", value=DEFAULT_METADATA_FILENAME)
    taxonomy_filename = st.sidebar.text_input("Taxonomy filename (optional)", value=DEFAULT_TAXONOMY_FILENAME)

    if not folder_id:
        st.warning("Add BOX_FOLDER_ID in secrets or enter it here.")
        st.stop()

    folder = _get_folder(client, folder_id)
    st.sidebar.success(f"Connected to: {folder.name} (auth: {auth_mode})")

    # Load data
    with st.spinner("Syncing metadata from Box..."):
        meta_df = load_metadata(client, folder_id, metadata_filename)
        tax_df = load_taxonomy(client, folder_id, taxonomy_filename)

    # Sync file IDs and shared URLs for any rows missing them
    items = _list_folder_items(client, folder_id)
    name_to_file = {i["name"]: i for i in items if i["type"] == "file"}
    changed = False
    for i, row in meta_df.iterrows():
        fname = str(row.get("file_name",""))
        if fname and (pd.isna(row.get("box_file_id")) or not str(row.get("box_file_id")).strip()):
            if fname in name_to_file:
                fid = name_to_file[fname]["id"]
                meta_df.at[i, "box_file_id"] = fid
                meta_df.at[i, "box_shared_url"] = _ensure_shared_link(client, fid)
                changed = True
    if changed:
        save_metadata(client, folder_id, metadata_filename, meta_df)

    # ---- Upload New Paper ----
    with st.expander("➕ Upload a new paper/slide (to Box)"):
        up = st.file_uploader("Upload PDF", type=["pdf"])
        colu1, colu2 = st.columns([2,1])
        with colu1:
            title = st.text_input("Title")
            authors = st.text_input("Authors (comma-separated)")
            year = st.text_input("Year")
            keywords = st.text_input("Keywords (comma/semicolon/pipe separated)  e.g., biosensor; aerosol; immunosensor")
            notes = st.text_area("Notes (optional)")
        with colu2:
            do_upload = st.button("Upload to Box & Append Metadata", use_container_width=True)

        if do_upload:
            if up is None:
                st.error("Please select a PDF.")
            elif up.size > PDF_MAX_MB*1024*1024:
                st.error(f"PDF too large. Max {PDF_MAX_MB} MB.")
            elif not title or not keywords:
                st.error("Please provide at least a Title and some Keywords.")
            else:
                # Upload to Box
                file_id = _upload_file_bytes(client, folder_id, up.name, up.read(), overwrite=False)
                if file_id:
                    url = _ensure_shared_link(client, file_id)
                    # Append row
                    new_row = {
                        "file_name": up.name, "title": title, "authors": authors,
                        "year": year, "keywords": keywords, "notes": notes,
                        "box_file_id": file_id, "box_shared_url": url
                    }
                    meta_df = pd.concat([meta_df, pd.DataFrame([new_row])], ignore_index=True)
                    save_metadata(client, folder_id, metadata_filename, meta_df)
                    st.success("Uploaded and metadata saved.")

    # ---- Edit Metadata ----
    with st.expander("✏️ Edit metadata table (saved to Box)", expanded=False):
        ed_df = st.data_editor(
            meta_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "keywords": st.column_config.TextColumn(help="Use separators: ',' ';' or '|'"),
                "box_shared_url": st.column_config.LinkColumn("box_shared_url")
            }
        )
        if st.button("Save changes to Box", type="primary"):
            save_metadata(client, folder_id, metadata_filename, ed_df)
            st.success("Metadata saved to Box.")
            meta_df = ed_df.copy()

    # ---- Filters ----
    st.subheader("Filters")
    c1, c2, c3, c4 = st.columns([1,1,1,1.2])
    with c1:
        year_f = st.text_input("Year contains", placeholder="e.g., 2021")
    with c2:
        author_f = st.text_input("Author contains", placeholder="e.g., Cui")
    with c3:
        kw_f = st.text_input("Keyword contains", placeholder="e.g., immunosensor")
    with c4:
        show_pdf = st.checkbox("Show PDF preview on select", value=True)

    # ---- Build graph ----
    with st.spinner("Building clusters..."):
        G_kw = build_keyword_graph(meta_df, tax_df)
        kw_partition = assign_keyword_communities(G_kw)
        net = build_pyvis_graph(meta_df, G_kw, kw_partition, {"year": year_f, "author": author_f, "search_kw": kw_f})

    # ---- Render graph ----
    st.subheader("Interactive Clustering Map")
    html_path = "graph.html"
    net.show(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        html_str = f.read()
    st.components.v1.html(html_str, height=780, scrolling=True)

    # ---- Document picker & preview ----
    st.markdown("### Browse Documents")
    opt = meta_df[["title","authors","year","file_name","box_file_id","box_shared_url","keywords"]].dropna(subset=["title"])
    opt["label"] = opt.apply(lambda r: f"{str(r['year']) if r['year'] else ''} — {r['title']} ({r['authors']})", axis=1)
    sel = st.selectbox("Select a document", ["(None)"] + opt["label"].tolist(), index=0)
    if sel != "(None)":
        row = opt[opt["label"] == sel].iloc[0]
        st.write(f"**Title:** {row['title']}")
        st.write(f"**Authors:** {row['authors']}")
        st.write(f"**Year:** {row['year']}")
        st.write(f"**Keywords:** {row['keywords']}")
        if row["box_shared_url"]:
            st.link_button("Open in Box", row["box_shared_url"])

        if show_pdf and row["box_file_id"]:
            pdf_bytes = _download_file_bytes(client, str(row["box_file_id"]))
            if pdf_bytes:
                st.markdown(pdf_bytes_to_iframe(pdf_bytes), unsafe_allow_html=True)
            st.download_button("Download PDF", data=pdf_bytes, file_name=row["file_name"], mime="application/pdf")

    # ---- Taxonomy help ----
    st.markdown("---")
    with st.expander("ℹ️ How taxonomy works (optional)"):
        st.markdown(
            """
            **taxonomy.csv** lets you define broader/narrower keywords and interlinks:

            - Columns: `child,parent`
            - Example:
              ```
              child,parent
              immunosensor,biosensor
              aptasensor,biosensor
              aerosol,environmental exposure
              ```
            These edges appear as **dashed** lines in the map. Co-occurrence edges are solid.
            """
        )


if __name__ == "__main__":
    main()
