#!/usr/bin/env python3
"""
Memory Palace (mempalace) — CLI app to interact with the ChromaDB knowledge base.

Commands:
    search <query>          Semantic search across the palace
    list                    List all wings and rooms with counts
    browse <wing> [room]    Browse entries in a wing/room
    add <text>              Add a new memory to the palace
    add-file <path>         Add a file's content to the palace
    delete <id>             Delete a memory by ID
    stats                   Show palace statistics
    export [wing] [room]    Export memories to JSON
"""

import argparse
import hashlib
import json
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import chromadb

PALACE_DIR = Path(__file__).parent
COLLECTION_NAME = "mempalace_drawers"
DEFAULT_WING = "claude_vnstock"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def get_client():
    return chromadb.PersistentClient(path=str(PALACE_DIR))


def get_collection(client=None):
    client = client or get_client()
    return client.get_or_create_collection(COLLECTION_NAME)


# ── helpers ──────────────────────────────────────────────────────────────────

def make_id(wing: str, room: str, text: str) -> str:
    h = hashlib.sha256(text.encode()).hexdigest()[:16]
    return f"drawer_{wing}_{room}_{h}"


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def truncate(text: str, width: int = 120) -> str:
    line = text.replace("\n", " ").strip()
    return line[:width] + "..." if len(line) > width else line


# ── commands ─────────────────────────────────────────────────────────────────

def _build_where(wing=None, room=None):
    """Build a ChromaDB where filter, using $and for multiple conditions."""
    conditions = []
    if wing:
        conditions.append({"wing": wing})
    if room:
        conditions.append({"room": room})
    if len(conditions) == 0:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def cmd_search(args):
    """Semantic search across the memory palace."""
    col = get_collection()
    query = " ".join(args.query)
    n = args.n or 10

    where = _build_where(args.wing, args.room)

    kwargs = dict(
        query_texts=[query],
        n_results=n,
        include=["metadatas", "documents", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = col.query(**kwargs)

    if not results["ids"][0]:
        print("No results found.")
        return

    print(f"\n{'─' * 80}")
    print(f"  Search: \"{query}\"  ({len(results['ids'][0])} results)")
    print(f"{'─' * 80}\n")

    for i, (doc_id, doc, meta, dist) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        similarity = max(0, 1 - dist / 2)  # L2 to approx similarity
        wing = meta.get("wing", "?")
        room = meta.get("room", "?")
        source = meta.get("source_file", "")
        filed = meta.get("filed_at", "")[:19]

        print(f"  [{i + 1}] score={similarity:.2f}  wing={wing}  room={room}")
        if source:
            print(f"      source: {Path(source).name}")
        if filed:
            print(f"      filed:  {filed}")
        print(f"      {truncate(doc, 100)}")
        print()

    if args.full and results["documents"][0]:
        print(f"\n{'═' * 80}")
        print("  Full content of top result:")
        print(f"{'═' * 80}\n")
        print(results["documents"][0][0])
        print()


def cmd_list(args):
    """List all wings and rooms with counts."""
    col = get_collection()
    total = col.count()

    # Fetch all metadata to aggregate
    batch_size = 5000
    wings_rooms = {}
    offset = 0

    while offset < total:
        batch = col.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )
        for meta in batch["metadatas"]:
            wing = meta.get("wing", "unknown")
            room = meta.get("room", "unknown")
            wings_rooms.setdefault(wing, {})
            wings_rooms[wing][room] = wings_rooms[wing].get(room, 0) + 1
        offset += batch_size

    print(f"\n  Memory Palace — {total:,} total memories\n")
    for wing in sorted(wings_rooms):
        wing_total = sum(wings_rooms[wing].values())
        print(f"  Wing: {wing}  ({wing_total:,} entries)")
        for room in sorted(wings_rooms[wing]):
            count = wings_rooms[wing][room]
            print(f"    └─ {room}: {count:,}")
        print()


def cmd_browse(args):
    """Browse entries in a wing/room."""
    col = get_collection()
    n = args.n or 20

    where = _build_where(args.wing, args.room)

    results = col.get(
        where=where or {"wing": args.wing},
        limit=n,
        offset=args.offset or 0,
        include=["metadatas", "documents"],
    )

    if not results["ids"]:
        print(f"No entries found in wing={args.wing}" +
              (f" room={args.room}" if args.room else ""))
        return

    label = f"wing={args.wing}"
    if args.room:
        label += f" room={args.room}"
    print(f"\n  Browsing {label}  (showing {len(results['ids'])} entries)\n")

    for doc_id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
        room = meta.get("room", "?")
        source = Path(meta.get("source_file", "?")).name
        filed = meta.get("filed_at", "")[:19]
        print(f"  [{doc_id[:40]}...]")
        print(f"    room={room}  source={source}  filed={filed}")
        print(f"    {truncate(doc, 100)}")
        print()


def cmd_add(args):
    """Add a new memory to the palace."""
    col = get_collection()
    text = " ".join(args.text)
    wing = args.wing or DEFAULT_WING
    room = args.room or "general"
    now = datetime.now(timezone.utc).isoformat()

    chunks = chunk_text(text)
    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        doc_id = make_id(wing, room, chunk)
        ids.append(doc_id)
        documents.append(chunk)
        metadatas.append({
            "wing": wing,
            "room": room,
            "added_by": "mempalace",
            "filed_at": now,
            "chunk_index": i,
            "source_file": args.source or "",
        })

    col.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Added {len(chunks)} chunk(s) to wing={wing} room={room}")


def cmd_add_file(args):
    """Add a file's content to the palace."""
    filepath = Path(args.path)
    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    text = filepath.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        print("File is empty.")
        sys.exit(1)

    col = get_collection()
    wing = args.wing or DEFAULT_WING
    room = args.room or "general"
    now = datetime.now(timezone.utc).isoformat()

    chunks = chunk_text(text)
    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        doc_id = make_id(wing, room, chunk)
        ids.append(doc_id)
        documents.append(chunk)
        metadatas.append({
            "wing": wing,
            "room": room,
            "added_by": "mempalace",
            "filed_at": now,
            "chunk_index": i,
            "source_file": str(filepath.resolve()),
        })

    col.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Added {len(chunks)} chunk(s) from {filepath.name} to wing={wing} room={room}")


def cmd_delete(args):
    """Delete a memory by ID or pattern."""
    col = get_collection()
    doc_id = args.id

    # Try exact match first
    existing = col.get(ids=[doc_id])
    if existing["ids"]:
        col.delete(ids=[doc_id])
        print(f"Deleted: {doc_id}")
        return

    # Try prefix match
    all_data = col.get(include=["metadatas"])
    matches = [i for i in all_data["ids"] if i.startswith(doc_id)]
    if not matches:
        print(f"No entry found matching: {doc_id}")
        return

    if len(matches) > 10:
        print(f"Found {len(matches)} matches — be more specific or use exact ID.")
        return

    col.delete(ids=matches)
    print(f"Deleted {len(matches)} entries matching '{doc_id}'")


def cmd_stats(args):
    """Show palace statistics."""
    col = get_collection()
    total = col.count()

    batch_size = 5000
    source_files = set()
    wings = set()
    rooms = set()
    earliest = None
    latest = None
    offset = 0

    while offset < total:
        batch = col.get(limit=batch_size, offset=offset, include=["metadatas"])
        for meta in batch["metadatas"]:
            wings.add(meta.get("wing", "?"))
            rooms.add(meta.get("room", "?"))
            sf = meta.get("source_file", "")
            if sf:
                source_files.add(sf)
            filed = meta.get("filed_at", "")
            if filed:
                if earliest is None or filed < earliest:
                    earliest = filed
                if latest is None or filed > latest:
                    latest = filed
        offset += batch_size

    db_size = PALACE_DIR / "chroma.sqlite3"
    size_mb = db_size.stat().st_size / (1024 * 1024) if db_size.exists() else 0

    print(f"\n  Memory Palace Statistics")
    print(f"  {'─' * 40}")
    print(f"  Total memories:   {total:,}")
    print(f"  Wings:            {len(wings)} ({', '.join(sorted(wings))})")
    print(f"  Rooms:            {len(rooms)} ({', '.join(sorted(rooms))})")
    print(f"  Source files:     {len(source_files)}")
    print(f"  Database size:    {size_mb:.1f} MB")
    if earliest:
        print(f"  Earliest entry:   {earliest[:19]}")
    if latest:
        print(f"  Latest entry:     {latest[:19]}")
    print()


def cmd_export(args):
    """Export memories to JSON."""
    col = get_collection()

    where = _build_where(args.wing, args.room)

    batch_size = 5000
    all_entries = []
    offset = 0

    while True:
        kwargs = dict(limit=batch_size, offset=offset, include=["metadatas", "documents"])
        if where:
            kwargs["where"] = where
        batch = col.get(**kwargs)
        if not batch["ids"]:
            break
        for doc_id, doc, meta in zip(batch["ids"], batch["documents"], batch["metadatas"]):
            all_entries.append({"id": doc_id, "document": doc, "metadata": meta})
        offset += batch_size

    output = args.output or "palace_export.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(all_entries)} entries to {output}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Memory Palace — interact with your ChromaDB knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python app.py search vnstock screener
              python app.py search -w claude_vnstock -r data_warehouse "parquet"
              python app.py list
              python app.py browse claude_vnstock
              python app.py browse claude_vnstock -r general -n 5
              python app.py add "Important note about VCB stock" -r data_warehouse
              python app.py add-file ./my_script.py -r general
              python app.py stats
              python app.py export -w claude_vnstock -o backup.json
        """),
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # search
    p_search = sub.add_parser("search", help="Semantic search across the palace")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("-n", type=int, default=10, help="Number of results (default: 10)")
    p_search.add_argument("-w", "--wing", help="Filter by wing")
    p_search.add_argument("-r", "--room", help="Filter by room")
    p_search.add_argument("--full", action="store_true", help="Show full content of top result")

    # list
    sub.add_parser("list", help="List all wings and rooms with counts")

    # browse
    p_browse = sub.add_parser("browse", help="Browse entries in a wing/room")
    p_browse.add_argument("wing", help="Wing name")
    p_browse.add_argument("-r", "--room", help="Filter by room")
    p_browse.add_argument("-n", type=int, default=20, help="Number of entries (default: 20)")
    p_browse.add_argument("--offset", type=int, default=0, help="Offset for pagination")

    # add
    p_add = sub.add_parser("add", help="Add a new memory")
    p_add.add_argument("text", nargs="+", help="Text content to add")
    p_add.add_argument("-w", "--wing", default=DEFAULT_WING, help="Wing name")
    p_add.add_argument("-r", "--room", default="general", help="Room name")
    p_add.add_argument("-s", "--source", default="", help="Source file reference")

    # add-file
    p_addfile = sub.add_parser("add-file", help="Add a file's content")
    p_addfile.add_argument("path", help="Path to the file")
    p_addfile.add_argument("-w", "--wing", default=DEFAULT_WING, help="Wing name")
    p_addfile.add_argument("-r", "--room", default="general", help="Room name")

    # delete
    p_delete = sub.add_parser("delete", help="Delete a memory by ID")
    p_delete.add_argument("id", help="Memory ID or prefix")

    # stats
    sub.add_parser("stats", help="Show palace statistics")

    # export
    p_export = sub.add_parser("export", help="Export memories to JSON")
    p_export.add_argument("-w", "--wing", help="Filter by wing")
    p_export.add_argument("-r", "--room", help="Filter by room")
    p_export.add_argument("-o", "--output", default="palace_export.json", help="Output file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "search": cmd_search,
        "list": cmd_list,
        "browse": cmd_browse,
        "add": cmd_add,
        "add-file": cmd_add_file,
        "delete": cmd_delete,
        "stats": cmd_stats,
        "export": cmd_export,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
