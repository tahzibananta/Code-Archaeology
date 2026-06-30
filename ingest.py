"""
Code Archaeology — Ingestion Pipeline

Pulls commits, pull requests, and issues from a GitHub repo and stores them
in Postgres with the foreign-key relationships needed to later answer
"why did this code change" queries.

Requires:
    pip install requests psycopg2-binary python-dotenv tenacity tree_sitter tree_sitter_languages

Env vars:
    GITHUB_TOKEN   - personal access token (repo scope, read-only is fine)
    DATABASE_URL   - postgres connection string

Usage:
    python ingest.py --repo owner/name --since 2023-01-01
"""

import os
import argparse
import time
from datetime import datetime, timezone
from typing import Optional

import requests
import psycopg2
import psycopg2.extras
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

GITHUB_API = "https://api.github.com"


# ---------------------------------------------------------------------------
# GitHub client
# ---------------------------------------------------------------------------

class GitHubClient:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def _get(self, url: str, params: Optional[dict] = None) -> requests.Response:
        resp = self.session.get(url, params=params)

        # Respect GitHub's rate limit headers rather than guessing
        if resp.status_code == 403 and "X-RateLimit-Remaining" in resp.headers:
            remaining = int(resp.headers.get("X-RateLimit-Remaining", "1"))
            if remaining == 0:
                reset_at = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                sleep_for = max(reset_at - time.time(), 1)
                print(f"Rate limited. Sleeping {sleep_for:.0f}s...")
                time.sleep(sleep_for)
                resp = self.session.get(url, params=params)

        resp.raise_for_status()
        return resp

    def paginate(self, url: str, params: Optional[dict] = None):
        """Yield items across all pages of a GitHub list endpoint."""
        params = dict(params or {})
        params.setdefault("per_page", 100)
        while url:
            resp = self._get(url, params=params)
            data = resp.json()
            if not isinstance(data, list):
                yield data
                return
            for item in data:
                yield item
            # GitHub paginates via Link headers, not page params after the first call
            url = resp.links.get("next", {}).get("url")
            params = None  # subsequent URL already encodes params

    def get_commits(self, owner: str, repo: str, since: Optional[str] = None):
        url = f"{GITHUB_API}/repos/{owner}/{repo}/commits"
        params = {"since": since} if since else {}
        return self.paginate(url, params)

    def get_commit_detail(self, owner: str, repo: str, sha: str) -> dict:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}"
        return self._get(url).json()

    def get_pulls(self, owner: str, repo: str, state: str = "all"):
        url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls"
        return self.paginate(url, {"state": state, "sort": "updated", "direction": "desc"})

    def get_pr_reviews(self, owner: str, repo: str, pr_number: int):
        url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
        return self.paginate(url)

    def get_pr_review_comments(self, owner: str, repo: str, pr_number: int):
        url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/comments"
        return self.paginate(url)

    def get_issues(self, owner: str, repo: str, state: str = "all"):
        # Note: this endpoint also returns PRs; we filter those out at call site
        url = f"{GITHUB_API}/repos/{owner}/{repo}/issues"
        return self.paginate(url, {"state": state, "sort": "updated", "direction": "desc"})

    def get_issue_comments(self, owner: str, repo: str, issue_number: int):
        url = f"{GITHUB_API}/repos/{owner}/{repo}/issues/{issue_number}/comments"
        return self.paginate(url)


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS commits (
    sha             TEXT PRIMARY KEY,
    repo            TEXT NOT NULL,
    author          TEXT,
    authored_at     TIMESTAMPTZ,
    message         TEXT,
    files_changed   JSONB,         -- [{filename, status, additions, deletions, patch}]
    pr_number       INTEGER        -- nullable; filled in via linking pass
);

CREATE TABLE IF NOT EXISTS pull_requests (
    pr_number       INTEGER,
    repo            TEXT NOT NULL,
    title           TEXT,
    body            TEXT,
    author          TEXT,
    state           TEXT,
    created_at      TIMESTAMPTZ,
    merged_at       TIMESTAMPTZ,
    merge_commit_sha TEXT,
    linked_issue_numbers INTEGER[],  -- parsed from body, e.g. "Fixes #123"
    PRIMARY KEY (repo, pr_number)
);

CREATE TABLE IF NOT EXISTS pr_comments (
    id              BIGINT PRIMARY KEY,
    repo            TEXT NOT NULL,
    pr_number       INTEGER,
    author          TEXT,
    body            TEXT,
    created_at      TIMESTAMPTZ,
    comment_type    TEXT  -- 'review' or 'review_comment'
);

CREATE TABLE IF NOT EXISTS issues (
    issue_number    INTEGER,
    repo            TEXT NOT NULL,
    title           TEXT,
    body            TEXT,
    author          TEXT,
    state           TEXT,
    created_at      TIMESTAMPTZ,
    closed_at       TIMESTAMPTZ,
    PRIMARY KEY (repo, issue_number)
);

CREATE TABLE IF NOT EXISTS issue_comments (
    id              BIGINT PRIMARY KEY,
    repo            TEXT NOT NULL,
    issue_number    INTEGER,
    author          TEXT,
    body            TEXT,
    created_at      TIMESTAMPTZ
);

-- Symbol-to-commit index, populated by a separate parsing pass (parse_symbols.py)
CREATE TABLE IF NOT EXISTS symbol_history (
    symbol          TEXT NOT NULL,
    repo            TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    sha             TEXT NOT NULL REFERENCES commits(sha),
    PRIMARY KEY (symbol, repo, file_path, sha)
);

CREATE INDEX IF NOT EXISTS idx_commits_repo ON commits(repo);
CREATE INDEX IF NOT EXISTS idx_pr_repo ON pull_requests(repo);
CREATE INDEX IF NOT EXISTS idx_issues_repo ON issues(repo);
CREATE INDEX IF NOT EXISTS idx_symbol_history_lookup ON symbol_history(repo, symbol);
"""


class Store:
    def __init__(self, dsn: str):
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = False

    def init_schema(self):
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA)
        self.conn.commit()

    def upsert_commit(self, repo: str, detail: dict):
        sha = detail["sha"]
        author = (detail.get("commit", {}).get("author") or {}).get("name")
        authored_at = (detail.get("commit", {}).get("author") or {}).get("date")
        message = detail.get("commit", {}).get("message")
        files = [
            {
                "filename": f.get("filename"),
                "status": f.get("status"),
                "additions": f.get("additions"),
                "deletions": f.get("deletions"),
                "patch": f.get("patch"),  # may be None for binary/large files
            }
            for f in detail.get("files", []) or []
        ]
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO commits (sha, repo, author, authored_at, message, files_changed)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (sha) DO UPDATE SET
                    files_changed = EXCLUDED.files_changed,
                    message = EXCLUDED.message
                """,
                (sha, repo, author, authored_at,
                 message, psycopg2.extras.Json(files)),
            )

    def upsert_pull_request(self, repo: str, pr: dict, linked_issues: list):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pull_requests
                    (pr_number, repo, title, body, author, state,
                     created_at, merged_at, merge_commit_sha, linked_issue_numbers)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (repo, pr_number) DO UPDATE SET
                    state = EXCLUDED.state,
                    merged_at = EXCLUDED.merged_at,
                    linked_issue_numbers = EXCLUDED.linked_issue_numbers
                """,
                (
                    pr["number"], repo, pr.get("title"), pr.get("body"),
                    (pr.get("user") or {}).get("login"), pr.get("state"),
                    pr.get("created_at"), pr.get("merged_at"),
                    pr.get("merge_commit_sha"), linked_issues,
                ),
            )

    def upsert_pr_comment(self, repo: str, pr_number: int, comment: dict, comment_type: str):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pr_comments (id, repo, pr_number, author, body, created_at, comment_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    comment["id"], repo, pr_number,
                    (comment.get("user") or {}).get("login"),
                    comment.get("body"), comment.get("created_at") or comment.get("submitted_at"),
                    comment_type,
                ),
            )

    def upsert_issue(self, repo: str, issue: dict):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO issues (issue_number, repo, title, body, author, state, created_at, closed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (repo, issue_number) DO UPDATE SET
                    state = EXCLUDED.state,
                    closed_at = EXCLUDED.closed_at
                """,
                (
                    issue["number"], repo, issue.get("title"), issue.get("body"),
                    (issue.get("user") or {}).get("login"), issue.get("state"),
                    issue.get("created_at"), issue.get("closed_at"),
                ),
            )

    def upsert_issue_comment(self, repo: str, issue_number: int, comment: dict):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO issue_comments (id, repo, issue_number, author, body, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    comment["id"], repo, issue_number,
                    (comment.get("user") or {}).get("login"),
                    comment.get("body"), comment.get("created_at"),
                ),
            )

    def commit(self):
        self.conn.commit()


# ---------------------------------------------------------------------------
# Linking helpers
# ---------------------------------------------------------------------------

import re

CLOSES_PATTERN = re.compile(
    r"\b(close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s*#(\d+)", re.IGNORECASE
)


def extract_linked_issues(pr_body: Optional[str]) -> list:
    """Parse 'Fixes #123' / 'Closes #45' style references out of a PR body."""
    if not pr_body:
        return []
    return [int(num) for _, num in CLOSES_PATTERN.findall(pr_body)]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_ingestion(owner: str, repo: str, since: Optional[str], dsn: str, token: str,
                   max_commits: Optional[int] = None, fetch_full_diffs: bool = True):
    repo_full = f"{owner}/{repo}"
    client = GitHubClient(token)
    store = Store(dsn)
    store.init_schema()

    # --- Commits ---
    print(f"Ingesting commits for {repo_full}...")
    count = 0
    for commit_summary in client.get_commits(owner, repo, since=since):
        sha = commit_summary["sha"]
        if fetch_full_diffs:
            # Per-commit call needed to get file-level diffs (list endpoint omits these)
            detail = client.get_commit_detail(owner, repo, sha)
        else:
            detail = commit_summary
        store.upsert_commit(repo_full, detail)
        count += 1
        if count % 50 == 0:
            store.commit()
            print(f"  ...{count} commits")
        if max_commits and count >= max_commits:
            break
    store.commit()
    print(f"Done: {count} commits.")

    # --- Pull requests + their discussion threads ---
    print(f"Ingesting pull requests for {repo_full}...")
    pr_count = 0
    for pr in client.get_pulls(owner, repo, state="all"):
        linked = extract_linked_issues(pr.get("body"))
        store.upsert_pull_request(repo_full, pr, linked)

        for review in client.get_pr_reviews(owner, repo, pr["number"]):
            if review.get("body"):  # skip empty approvals
                store.upsert_pr_comment(repo_full, pr["number"], review, "review")

        for comment in client.get_pr_review_comments(owner, repo, pr["number"]):
            store.upsert_pr_comment(repo_full, pr["number"], comment, "review_comment")

        pr_count += 1
        if pr_count % 25 == 0:
            store.commit()
            print(f"  ...{pr_count} PRs")
    store.commit()
    print(f"Done: {pr_count} pull requests.")

    # --- Issues + comments (skip items that are actually PRs) ---
    print(f"Ingesting issues for {repo_full}...")
    issue_count = 0
    for issue in client.get_issues(owner, repo, state="all"):
        if "pull_request" in issue:
            continue  # GitHub's issues endpoint includes PRs; skip duplicates
        store.upsert_issue(repo_full, issue)
        for comment in client.get_issue_comments(owner, repo, issue["number"]):
            store.upsert_issue_comment(repo_full, issue["number"], comment)
        issue_count += 1
        if issue_count % 25 == 0:
            store.commit()
            print(f"  ...{issue_count} issues")
    store.commit()
    print(f"Done: {issue_count} issues.")


def main():
    parser = argparse.ArgumentParser(description="Ingest a GitHub repo for Code Archaeology")
    parser.add_argument("--repo", required=True, help="owner/name, e.g. psf/requests")
    parser.add_argument("--since", default=None, help="ISO date, e.g. 2023-01-01")
    parser.add_argument("--max-commits", type=int, default=None)
    parser.add_argument("--no-full-diffs", action="store_true",
                         help="Skip per-commit detail calls (faster, but no patch text)")
    args = parser.parse_args()

    owner, name = args.repo.split("/")
    token = os.environ["GITHUB_TOKEN"]
    dsn = os.environ["DATABASE_URL"]

    since_iso = None
    if args.since:
        since_iso = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc).isoformat()

    run_ingestion(
        owner, name, since_iso, dsn, token,
        max_commits=args.max_commits,
        fetch_full_diffs=not args.no_full_diffs,
    )


if __name__ == "__main__":
    main()
