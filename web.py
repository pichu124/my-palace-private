"""
Memory Palace — Streamlit Web UI
Provides a browser-based interface to the ChromaDB knowledge base.
"""

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import streamlit as st

PALACE_DIR = Path(__file__).parent
COLLECTION_NAME = "mempalace_drawers"
DEFAULT_WING = "claude_vnstock"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Memory Palace",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── ChromaDB client (cached) ──────────────────────────────────────────────────

@st.cache_resource
def get_client():
    return chromadb.PersistentClient(path=str(PALACE_DIR))


@st.cache_resource
def get_collection():
    client = get_client()
    return client.get_or_create_collection(COLLECTION_NAME)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_where(wing: str | None, room: str | None):
    """Build a ChromaDB where filter; use $and for multiple conditions."""
    conditions = []
    if wing:
        conditions.append({"wing": wing})
    if room:
        conditions.append({"room": room})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def load_wing_room_map() -> dict[str, list[str]]:
    """Return {wing: [room, ...]} by scanning all metadata."""
    col = get_collection()
    total = col.count()
    mapping: dict[str, set] = {}
    offset = 0
    while offset < total:
        batch = col.get(limit=5000, offset=offset, include=["metadatas"])
        for meta in batch["metadatas"]:
            w = meta.get("wing", "unknown")
            r = meta.get("room", "unknown")
            mapping.setdefault(w, set()).add(r)
        offset += 5000
    return {w: sorted(mapping[w]) for w in sorted(mapping)}


def make_id(wing: str, room: str, text: str) -> str:
    h = hashlib.sha256(text.encode()).hexdigest()[:16]
    return f"drawer_{wing}_{room}_{h}"


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def score_from_distance(dist: float) -> float:
    """Convert L2 distance to approximate similarity [0,1]."""
    return max(0.0, 1.0 - dist / 2.0)


# ── Tabs ──────────────────────────────────────────────────────────────────────

st.title("🏛️ Memory Palace")

tab_ask, tab_search, tab_browse, tab_stats, tab_add = st.tabs([
    "💬 Ask",
    "🔍 Search",
    "📚 Browse",
    "📊 Stats",
    "➕ Add Memory",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ask (RAG Q&A)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ask:
    st.subheader("Ask a Question")
    st.caption("Retrieves relevant context from the knowledge base and asks Claude.")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.warning(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "The Ask tab requires it to call Claude."
        )

    # Filters
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        wing_map_ask = load_wing_room_map()
        wings_ask = ["(all wings)"] + list(wing_map_ask.keys())
        selected_wing_ask = st.selectbox("Filter by wing", wings_ask, key="ask_wing")
        wing_filter_ask = None if selected_wing_ask == "(all wings)" else selected_wing_ask

    with col_w2:
        if wing_filter_ask and wing_map_ask.get(wing_filter_ask):
            rooms_ask = ["(all rooms)"] + wing_map_ask[wing_filter_ask]
        else:
            all_rooms_ask: set[str] = set()
            for rooms in wing_map_ask.values():
                all_rooms_ask.update(rooms)
            rooms_ask = ["(all rooms)"] + sorted(all_rooms_ask)
        selected_room_ask = st.selectbox("Filter by room", rooms_ask, key="ask_room")
        room_filter_ask = None if selected_room_ask == "(all rooms)" else selected_room_ask

    question = st.text_input(
        "Your question",
        placeholder="e.g. How do I filter stocks by P/E ratio in vnstock?",
        key="ask_question",
    )

    if st.button("Ask Claude", type="primary", key="ask_btn"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not api_key:
            st.error("ANTHROPIC_API_KEY is not set. Cannot call Claude.")
        else:
            col = get_collection()
            where = build_where(wing_filter_ask, room_filter_ask)

            with st.spinner("Retrieving context from Memory Palace..."):
                kwargs = dict(
                    query_texts=[question],
                    n_results=6,
                    include=["metadatas", "documents", "distances"],
                )
                if where:
                    kwargs["where"] = where
                results = col.query(**kwargs)

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            if not docs:
                st.info("No relevant context found in the knowledge base.")
            else:
                context_parts = []
                for i, (doc, meta) in enumerate(zip(docs, metas)):
                    source = Path(meta.get("source_file", "")).name or "unknown"
                    room_lbl = meta.get("room", "?")
                    wing_lbl = meta.get("wing", "?")
                    context_parts.append(
                        f"[Chunk {i+1} | wing={wing_lbl} room={room_lbl} source={source}]\n{doc}"
                    )
                context_text = "\n\n---\n\n".join(context_parts)

                system_prompt = (
                    "You are a helpful assistant with access to a personal knowledge base "
                    "called Memory Palace. Answer the user's question using ONLY the provided "
                    "context chunks. If the answer cannot be found in the context, say so clearly. "
                    "The context may contain Vietnamese text — respond in the same language as the question."
                )
                user_message = (
                    f"Context from Memory Palace:\n\n{context_text}\n\n"
                    f"---\n\nQuestion: {question}"
                )

                with st.spinner("Asking Claude..."):
                    try:
                        import anthropic
                        client_ai = anthropic.Anthropic(api_key=api_key)
                        response = client_ai.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=2048,
                            system=system_prompt,
                            messages=[{"role": "user", "content": user_message}],
                        )
                        answer = response.content[0].text
                    except Exception as e:
                        st.error(f"Claude API error: {e}")
                        answer = None

                if answer:
                    with st.chat_message("assistant"):
                        st.markdown(answer)

                    with st.expander(
                        f"📎 Retrieved context ({len(docs)} chunks)", expanded=False
                    ):
                        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
                            score = score_from_distance(dist)
                            source = Path(meta.get("source_file", "")).name or "—"
                            filed = meta.get("filed_at", "")[:19] or "—"
                            wing_lbl = meta.get("wing", "?")
                            room_lbl = meta.get("room", "?")

                            st.markdown(
                                f"**Chunk {i+1}** &nbsp;|&nbsp; "
                                f"score `{score:.2f}` &nbsp;|&nbsp; "
                                f"wing `{wing_lbl}` &nbsp;|&nbsp; "
                                f"room `{room_lbl}` &nbsp;|&nbsp; "
                                f"source `{source}` &nbsp;|&nbsp; "
                                f"filed `{filed}`"
                            )
                            st.text(doc[:800] + ("..." if len(doc) > 800 else ""))
                            st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Search
# ═══════════════════════════════════════════════════════════════════════════════

with tab_search:
    st.subheader("Semantic Search")
    st.caption("Find the most relevant chunks in the knowledge base.")

    # Filters row
    col_s1, col_s2, col_s3 = st.columns([3, 2, 2])
    with col_s1:
        search_query = st.text_input(
            "Search query",
            placeholder="e.g. lọc cổ phiếu theo vốn hóa",
            key="search_query",
        )
    with col_s2:
        wing_map_search = load_wing_room_map()
        wings_search = ["(all wings)"] + list(wing_map_search.keys())
        selected_wing_search = st.selectbox("Wing", wings_search, key="search_wing")
        wing_filter_search = (
            None if selected_wing_search == "(all wings)" else selected_wing_search
        )
    with col_s3:
        if wing_filter_search and wing_map_search.get(wing_filter_search):
            rooms_search = ["(all rooms)"] + wing_map_search[wing_filter_search]
        else:
            all_rooms_s: set[str] = set()
            for rms in wing_map_search.values():
                all_rooms_s.update(rms)
            rooms_search = ["(all rooms)"] + sorted(all_rooms_s)
        selected_room_search = st.selectbox("Room", rooms_search, key="search_room")
        room_filter_search = (
            None if selected_room_search == "(all rooms)" else selected_room_search
        )

    n_results = st.slider("Number of results", min_value=1, max_value=20, value=10, key="search_n")

    if st.button("Search", type="primary", key="search_btn"):
        if not search_query.strip():
            st.warning("Please enter a search query.")
        else:
            col = get_collection()
            where = build_where(wing_filter_search, room_filter_search)

            with st.spinner("Searching..."):
                kwargs = dict(
                    query_texts=[search_query],
                    n_results=n_results,
                    include=["metadatas", "documents", "distances"],
                )
                if where:
                    kwargs["where"] = where
                results = col.query(**kwargs)

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            if not docs:
                st.info("No results found.")
            else:
                st.success(f"Found {len(docs)} result(s) for: **{search_query}**")
                for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
                    score = score_from_distance(dist)
                    source = Path(meta.get("source_file", "")).name or "—"
                    filed = meta.get("filed_at", "")[:19] or "—"
                    wing_lbl = meta.get("wing", "?")
                    room_lbl = meta.get("room", "?")
                    chunk_idx = meta.get("chunk_index", "?")

                    header = (
                        f"#{i+1} — score `{score:.3f}` | "
                        f"wing `{wing_lbl}` | room `{room_lbl}` | "
                        f"source `{source}`"
                    )
                    with st.expander(header, expanded=(i < 3)):
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Similarity Score", f"{score:.3f}")
                        col_b.metric("Chunk Index", chunk_idx)
                        col_c.metric("Filed At", filed[:10] if filed != "—" else "—")
                        st.markdown(f"**Source file:** `{source}`")
                        st.markdown("**Content:**")
                        st.text(doc)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Browse
# ═══════════════════════════════════════════════════════════════════════════════

with tab_browse:
    st.subheader("Browse Knowledge Base")
    st.caption("Browse entries by wing and room with pagination.")

    wing_map_browse = load_wing_room_map()

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        wings_browse = list(wing_map_browse.keys())
        if not wings_browse:
            st.info("No wings found in the database.")
            st.stop()
        selected_wing_browse = st.selectbox("Wing", wings_browse, key="browse_wing")

    with col_b2:
        rooms_browse = ["(all rooms)"] + wing_map_browse.get(selected_wing_browse, [])
        selected_room_browse = st.selectbox("Room", rooms_browse, key="browse_room")
        room_filter_browse = (
            None if selected_room_browse == "(all rooms)" else selected_room_browse
        )

    PAGE_SIZE = 20
    page_num = st.number_input("Page", min_value=1, value=1, step=1, key="browse_page")
    offset = (page_num - 1) * PAGE_SIZE

    where_browse = build_where(selected_wing_browse, room_filter_browse)

    col = get_collection()
    with st.spinner("Loading entries..."):
        kwargs_browse = dict(
            limit=PAGE_SIZE,
            offset=offset,
            include=["metadatas", "documents"],
        )
        if where_browse:
            kwargs_browse["where"] = where_browse
        else:
            kwargs_browse["where"] = {"wing": selected_wing_browse}

        browse_results = col.get(**kwargs_browse)

    ids = browse_results["ids"]
    docs_browse = browse_results["documents"]
    metas_browse = browse_results["metadatas"]

    if not ids:
        st.info(f"No entries found (page {page_num}).")
    else:
        label = f"wing=`{selected_wing_browse}`"
        if room_filter_browse:
            label += f" room=`{room_filter_browse}`"
        st.markdown(
            f"Showing **{len(ids)}** entries for {label} "
            f"(page {page_num}, offset {offset})"
        )

        for doc_id, doc, meta in zip(ids, docs_browse, metas_browse):
            room_lbl = meta.get("room", "?")
            source = Path(meta.get("source_file", "")).name or "—"
            filed = meta.get("filed_at", "")[:19] or "—"
            chunk_idx = meta.get("chunk_index", "?")
            added_by = meta.get("added_by", "—")

            preview = doc[:120].replace("\n", " ") + ("..." if len(doc) > 120 else "")
            header = f"`{room_lbl}` — {preview}"

            with st.expander(header):
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.markdown(f"**Room**\n\n`{room_lbl}`")
                col_m2.markdown(f"**Source**\n\n`{source}`")
                col_m3.markdown(f"**Filed**\n\n`{filed}`")
                col_m4.markdown(f"**Chunk #**\n\n`{chunk_idx}`")
                st.markdown(f"**ID:** `{doc_id}`")
                st.markdown("**Content:**")
                st.text(doc)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Stats
# ═══════════════════════════════════════════════════════════════════════════════

with tab_stats:
    st.subheader("Knowledge Base Statistics")

    col = get_collection()

    with st.spinner("Computing statistics..."):
        total = col.count()
        batch_size = 5000
        source_files: set[str] = set()
        wings_seen: set[str] = set()
        rooms_seen: set[str] = set()
        wing_room_counts: dict[str, dict[str, int]] = {}
        earliest = None
        latest = None
        offset_stats = 0

        while offset_stats < total:
            batch = col.get(limit=batch_size, offset=offset_stats, include=["metadatas"])
            for meta in batch["metadatas"]:
                w = meta.get("wing", "unknown")
                r = meta.get("room", "unknown")
                wings_seen.add(w)
                rooms_seen.add(r)
                wing_room_counts.setdefault(w, {})
                wing_room_counts[w][r] = wing_room_counts[w].get(r, 0) + 1
                sf = meta.get("source_file", "")
                if sf:
                    source_files.add(sf)
                filed = meta.get("filed_at", "")
                if filed:
                    if earliest is None or filed < earliest:
                        earliest = filed
                    if latest is None or filed > latest:
                        latest = filed
            offset_stats += batch_size

    db_path = PALACE_DIR / "chroma.sqlite3"
    size_mb = db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0

    # Top metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Memories", f"{total:,}")
    m2.metric("Wings", len(wings_seen))
    m3.metric("Rooms", len(rooms_seen))
    m4.metric("Source Files", len(source_files))
    m5.metric("DB Size", f"{size_mb:.1f} MB")

    if earliest or latest:
        c1, c2 = st.columns(2)
        if earliest:
            c1.metric("Earliest Entry", earliest[:10])
        if latest:
            c2.metric("Latest Entry", latest[:10])

    st.divider()
    st.subheader("Breakdown by Wing / Room")

    import pandas as pd

    rows = []
    for w in sorted(wing_room_counts):
        for r in sorted(wing_room_counts[w]):
            rows.append({
                "Wing": w,
                "Room": r,
                "Count": wing_room_counts[w][r],
            })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)

        st.divider()
        st.subheader("Wing Totals")
        wing_totals = df.groupby("Wing")["Count"].sum().reset_index()
        wing_totals.columns = ["Wing", "Total"]
        st.bar_chart(wing_totals.set_index("Wing")["Total"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Add Memory
# ═══════════════════════════════════════════════════════════════════════════════

with tab_add:
    st.subheader("Add a New Memory")
    st.caption("Chunks and embeds the text into the ChromaDB knowledge base.")

    wing_map_add = load_wing_room_map()
    existing_wings = list(wing_map_add.keys())

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        wing_input = st.text_input(
            "Wing",
            value=DEFAULT_WING,
            placeholder="e.g. claude_vnstock",
            key="add_wing",
        )
        if existing_wings:
            st.caption(
                "Existing wings: " + ", ".join(f"`{w}`" for w in existing_wings)
            )
    with col_a2:
        room_input = st.text_input(
            "Room",
            value="general",
            placeholder="e.g. general, data_warehouse",
            key="add_room",
        )
        if wing_input and wing_map_add.get(wing_input):
            st.caption(
                "Existing rooms in this wing: "
                + ", ".join(f"`{r}`" for r in wing_map_add[wing_input])
            )

    source_input = st.text_input(
        "Source file (optional)",
        placeholder="e.g. my_notes.txt or /path/to/file.py",
        key="add_source",
    )

    content_input = st.text_area(
        "Content",
        placeholder="Paste or type the text you want to store...",
        height=300,
        key="add_content",
    )

    st.caption(
        f"Text longer than {CHUNK_SIZE} characters will be split into overlapping chunks "
        f"(chunk size {CHUNK_SIZE}, overlap {CHUNK_OVERLAP})."
    )

    if st.button("Add to Memory Palace", type="primary", key="add_btn"):
        if not content_input.strip():
            st.warning("Please enter some content.")
        elif not wing_input.strip():
            st.warning("Wing cannot be empty.")
        elif not room_input.strip():
            st.warning("Room cannot be empty.")
        else:
            col_db = get_collection()
            now = datetime.now(timezone.utc).isoformat()
            chunks = chunk_text(content_input.strip())

            ids_to_add = []
            docs_to_add = []
            metas_to_add = []

            for i, chunk in enumerate(chunks):
                doc_id = make_id(wing_input.strip(), room_input.strip(), chunk)
                ids_to_add.append(doc_id)
                docs_to_add.append(chunk)
                metas_to_add.append({
                    "wing": wing_input.strip(),
                    "room": room_input.strip(),
                    "added_by": "web_ui",
                    "filed_at": now,
                    "chunk_index": i,
                    "source_file": source_input.strip(),
                })

            with st.spinner(f"Adding {len(chunks)} chunk(s) to the palace..."):
                try:
                    col_db.add(
                        ids=ids_to_add,
                        documents=docs_to_add,
                        metadatas=metas_to_add,
                    )
                    # Invalidate the cache so wing/room maps refresh
                    get_collection.clear()
                    st.success(
                        f"Successfully added **{len(chunks)} chunk(s)** to "
                        f"wing=`{wing_input.strip()}` room=`{room_input.strip()}`."
                    )
                    st.info("Refresh the page to see updated stats and browse the new entries.")
                except Exception as e:
                    st.error(f"Failed to add memory: {e}")
