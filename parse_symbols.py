"""
Code Archaeology — Symbol Parsing Pass

Reads the commits already ingested by ingest.py, figures out which
functions/classes each commit's diff actually touched, and writes that
mapping into symbol_history. This is what lets the agent later answer
"show me every commit that touched parse_config()" instead of relying
on keyword search alone.

Approach:
    1. For each commit, parse the unified diff hunk headers (@@ -a,b +c,d @@)
       to get the changed line ranges in the post-change file.
    2. Fetch the file's full content as of that commit (via GitHub API).
    3. Parse that file with tree-sitter to get every function/class symbol
       and its line span.
    4. Any symbol whose span overlaps a changed range gets a row in
       symbol_history.

Requires:
    pip install tree_sitter tree_sitter_languages psycopg2-binary requests tenacity

Env vars:
    GITHUB_TOKEN, DATABASE_URL  (same as ingest.py)

Usage:
    python parse_symbols.py --repo owner/name
"""

import os
import re
import argparse
import base64
from typing import Optional

import requests
import psycopg2
import psycopg2.extras
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tree_sitter_languages import get_parser

GITHUB_API = "https://api.github.com"

# Map file extensions to tree-sitter language names and the node types
# that count as a "symbol" worth indexing.
LANGUAGE_CONFIG = {
    ".py": {
        "lang": "python",
        "symbol_nodes": {"function_definition", "class_definition"},
        "name_field": "name",
    },
    ".js": {
        "lang": "javascript",
        "symbol_nodes": {"function_declaration", "class_declaration", "method_definition"},
        "name_field": "name",
    },
    ".ts": {
        "lang": "typescript",
        "symbol_nodes": {"function_declaration", "class_declaration", "method_definition"},
        "name_field": "name",
    },
    ".go": {
        "lang": "go",
        "symbol_nodes": {"function_declaration", "method_declaration"},
        "name_field": "name",
    },
    ".java": {
        "lang": "java",
        "symbol_nodes": {"method_declaration", "class_declaration"},
        "name_field": "name",
    },
    ".rb": {
        "lang": "ruby",
        "symbol_nodes": {"method", "class"},
        "name_field": "name",
    },
}

HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


# ---------------------------------------------------------------------------
# Diff parsing — figure out which lines actually changed
# ---------------------------------------------------------------------------

def changed_line_ranges(patch: Optional[str]) -> list:
    """
    Parse a GitHub unified diff 'patch' string and return the list of
    (start_line, end_line) ranges in the POST-change file that were
    added or modified. Pure deletions don't map to post-change lines
    and are skipped, since there's no surviving symbol to attribute them to.
    """
    if not patch:
        return []

    ranges = []
    current_line = None
    for line in patch.splitlines():
        m = HUNK_HEADER.match(line)
        if m:
            current_line = int(m.group(1))
            continue
        if current_line is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            ranges.append((current_line, current_line))
            current_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            pass  # deletion; doesn't advance post-change line counter
        else:
            current_line += 1

    return _merge_ranges(ranges)


def _merge_ranges(ranges: list) -> list:
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


# ---------------------------------------------------------------------------
# GitHub file content fetch
# ---------------------------------------------------------------------------

class GitHubFileFetcher:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        })
        self._cache = {}  # (sha, path) -> content, avoids refetching same blob

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def get_file_at_commit(self, owner: str, repo: str, path: str, sha: str) -> Optional[str]:
        key = (sha, path)
        if key in self._cache:
            return self._cache[key]

        url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
        resp = self.session.get(url, params={"ref": sha})
        if resp.status_code == 404:
            self._cache[key] = None
            return None
        resp.raise_for_status()
        data = resp.json()
        if data.get("encoding") != "base64":
            self._cache[key] = None
            return None
        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        self._cache[key] = content
        return content


# ---------------------------------------------------------------------------
# Tree-sitter symbol extraction
# ---------------------------------------------------------------------------

_parser_cache = {}


def get_cached_parser(lang: str):
    if lang not in _parser_cache:
        _parser_cache[lang] = get_parser(lang)
    return _parser_cache[lang]


def extract_symbols(source: str, ext: str) -> list:
    """
    Return a list of (symbol_name, start_line, end_line) for every
    function/class in the file, using 1-indexed lines to match diff
    hunk numbering.
    """
    config = LANGUAGE_CONFIG.get(ext)
    if not config:
        return []

    parser = get_cached_parser(config["lang"])
    tree = parser.parse(source.encode("utf-8"))
    symbols = []

    def walk(node):
        if node.type in config["symbol_nodes"]:
            name_node = node.child_by_field_name(config["name_field"])
            name = name_node.text.decode("utf-8") if name_node else "<anonymous>"
            start_line = node.start_point[0] + 1  # tree-sitter is 0-indexed
            end_line = node.end_point[0] + 1
            symbols.append((name, start_line, end_line))
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return symbols


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_symbol_pass(owner: str, repo: str, dsn: str, token: str, limit: Optional[int] = None):
    repo_full = f"{owner}/{repo}"
    fetcher = GitHubFileFetcher(token)
    conn = psycopg2.connect(dsn)
    conn.autocommit = False

    with conn.cursor(name="commit_cursor") as cur:  # server-side cursor: repos can have huge history
        cur.execute(
            "SELECT sha, files_changed FROM commits WHERE repo = %s ORDER BY authored_at",
            (repo_full,),
        )

        processed = 0
        symbol_rows = 0
        write_cur = conn.cursor()

        for sha, files_changed in cur:
            for f in (files_changed or []):
                filename = f.get("filename")
                patch = f.get("patch")
                status = f.get("status")
                if not filename or status == "removed":
                    continue

                ext = os.path.splitext(filename)[1]
                if ext not in LANGUAGE_CONFIG:
                    continue  # skip unsupported languages, binary files, configs, etc.

                changed_ranges = changed_line_ranges(patch)
                if not changed_ranges:
                    continue

                source = fetcher.get_file_at_commit(owner, repo, filename, sha)
                if source is None:
                    continue

                try:
                    symbols = extract_symbols(source, ext)
                except Exception as e:
                    # Malformed/unparseable source shouldn't kill the whole run
                    print(f"  [warn] failed to parse {filename}@{sha}: {e}")
                    continue

                touched_symbols = {
                    name for name, s_start, s_end in symbols
                    if any(ranges_overlap(s_start, s_end, c_start, c_end)
                           for c_start, c_end in changed_ranges)
                }

                for symbol_name in touched_symbols:
                    write_cur.execute(
                        """
                        INSERT INTO symbol_history (symbol, repo, file_path, sha)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (symbol_name, repo_full, filename, sha),
                    )
                    symbol_rows += 1

            processed += 1
            if processed % 25 == 0:
                conn.commit()
                print(f"  ...{processed} commits processed, {symbol_rows} symbol links written")
            if limit and processed >= limit:
                break

        conn.commit()
        print(f"Done: {processed} commits processed, {symbol_rows} symbol links written.")


def main():
    parser = argparse.ArgumentParser(description="Build symbol -> commit history index")
    parser.add_argument("--repo", required=True, help="owner/name, e.g. psf/requests")
    parser.add_argument("--limit", type=int, default=None, help="cap number of commits processed")
    args = parser.parse_args()

    owner, name = args.repo.split("/")
    token = os.environ["GITHUB_TOKEN"]
    dsn = os.environ["DATABASE_URL"]

    run_symbol_pass(owner, name, dsn, token, limit=args.limit)


if __name__ == "__main__":
    main()
