"""
Code Archaeology — Agent Layer

The reasoning core of the project. Given a natural-language question about
a symbol or piece of code ("why does parse_config() validate twice?"), the
agent decides which tools to call, chains them (commit -> PR -> issue), and
synthesizes a grounded, cited answer.

This is intentionally NOT single-shot RAG: the model is given tools and
makes its own decisions about what to look up, in what order, mirroring
how a human engineer would investigate.

Requires:
    pip install anthropic psycopg2-binary

Env vars:
    ANTHROPIC_API_KEY, DATABASE_URL

Usage:
    python agent.py --repo owner/name --question "Why was retry logic added to the HTTP client?"
"""

import os
import json
import argparse
from typing import Optional

import psycopg2
import psycopg2.extras
import anthropic

MODEL = "claude-sonnet-4-6"
MAX_TOOL_ITERATIONS = 8  # hard cap so a confused agent can't loop forever


# ---------------------------------------------------------------------------
# Tool implementations — thin wrappers over Postgres
# ---------------------------------------------------------------------------

class ArchaeologyTools:
    def __init__(self, dsn: str, repo: str):
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True
        self.repo = repo

    def get_commit_history(self, symbol: str, limit: int = 15) -> list:
        """All commits that touched a given function/class, newest first."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT c.sha, c.author, c.authored_at, c.message, c.pr_number
                FROM symbol_history sh
                JOIN commits c ON c.sha = sh.sha
                WHERE sh.repo = %s AND sh.symbol = %s
                ORDER BY c.authored_at DESC
                LIMIT %s
                """,
                (self.repo, symbol, limit),
            )
            rows = cur.fetchall()
        for r in rows:
            r["authored_at"] = r["authored_at"].isoformat() if r["authored_at"] else None
        return rows

    def get_diff(self, sha: str, file_path: Optional[str] = None) -> list:
        """The actual patch text for a commit, optionally filtered to one file."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT files_changed, message FROM commits WHERE repo = %s AND sha = %s",
                (self.repo, sha),
            )
            row = cur.fetchone()
        if not row:
            return []
        files = row["files_changed"] or []
        if file_path:
            files = [f for f in files if f.get("filename") == file_path]
        # Truncate huge patches so we don't blow the context window on one file
        for f in files:
            if f.get("patch") and len(f["patch"]) > 4000:
                f["patch"] = f["patch"][:4000] + "\n... [truncated]"
        return files

    def get_pr_discussion(self, pr_number: int) -> dict:
        """A PR's description plus its review comments, and any linked issues."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT title, body, author, state, created_at, merged_at, linked_issue_numbers
                FROM pull_requests WHERE repo = %s AND pr_number = %s
                """,
                (self.repo, pr_number),
            )
            pr = cur.fetchone()
            if not pr:
                return {}

            cur.execute(
                """
                SELECT author, body, created_at, comment_type
                FROM pr_comments WHERE repo = %s AND pr_number = %s
                ORDER BY created_at
                """,
                (self.repo, pr_number),
            )
            comments = cur.fetchall()

        pr["created_at"] = pr["created_at"].isoformat() if pr["created_at"] else None
        pr["merged_at"] = pr["merged_at"].isoformat() if pr["merged_at"] else None
        for c in comments:
            c["created_at"] = c["created_at"].isoformat() if c["created_at"] else None

        return {"pr": pr, "comments": comments}

    def get_issue(self, issue_number: int) -> dict:
        """An issue's body plus its comment thread."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT title, body, author, state, created_at, closed_at
                FROM issues WHERE repo = %s AND issue_number = %s
                """,
                (self.repo, issue_number),
            )
            issue = cur.fetchone()
            if not issue:
                return {}
            cur.execute(
                """
                SELECT author, body, created_at
                FROM issue_comments WHERE repo = %s AND issue_number = %s
                ORDER BY created_at
                """,
                (self.repo, issue_number),
            )
            comments = cur.fetchall()

        issue["created_at"] = issue["created_at"].isoformat() if issue["created_at"] else None
        issue["closed_at"] = issue["closed_at"].isoformat() if issue["closed_at"] else None
        for c in comments:
            c["created_at"] = c["created_at"].isoformat() if c["created_at"] else None

        return {"issue": issue, "comments": comments}

    def search_issues(self, query: str, limit: int = 10) -> list:
        """
        Full-text search across issue titles/bodies and PR titles/bodies.
        Uses Postgres's built-in tsvector/plainto_tsquery rather than a
        separate vector DB, to keep infra minimal -- swap for pgvector
        embeddings later if recall on paraphrased queries becomes an issue.
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                (
                    SELECT 'issue' AS kind, issue_number AS number, title, state,
                           ts_rank(to_tsvector('english', coalesce(title,'') || ' ' || coalesce(body,'')),
                                    plainto_tsquery('english', %s)) AS rank
                    FROM issues
                    WHERE repo = %s
                      AND to_tsvector('english', coalesce(title,'') || ' ' || coalesce(body,''))
                          @@ plainto_tsquery('english', %s)
                )
                UNION ALL
                (
                    SELECT 'pr' AS kind, pr_number AS number, title, state,
                           ts_rank(to_tsvector('english', coalesce(title,'') || ' ' || coalesce(body,'')),
                                    plainto_tsquery('english', %s)) AS rank
                    FROM pull_requests
                    WHERE repo = %s
                      AND to_tsvector('english', coalesce(title,'') || ' ' || coalesce(body,''))
                          @@ plainto_tsquery('english', %s)
                )
                ORDER BY rank DESC
                LIMIT %s
                """,
                (query, self.repo, query, query, self.repo, query, limit),
            )
            return cur.fetchall()


# ---------------------------------------------------------------------------
# Tool schema (Anthropic tool-use format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "get_commit_history",
        "description": "Get every commit that touched a specific function or class, newest first. Use this first when the question names a specific symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Exact function or class name, e.g. 'parse_config'"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_diff",
        "description": "Get the actual code patch for a commit. Use this to see exactly what changed, after get_commit_history has identified a candidate commit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sha": {"type": "string"},
                "file_path": {"type": "string", "description": "Optional: limit to one file if the commit touched many"},
            },
            "required": ["sha"],
        },
    },
    {
        "name": "get_pr_discussion",
        "description": "Get a pull request's description and review comments. Use this when a commit has a pr_number, to find the human reasoning behind the change.",
        "input_schema": {
            "type": "object",
            "properties": {"pr_number": {"type": "integer"}},
            "required": ["pr_number"],
        },
    },
    {
        "name": "get_issue",
        "description": "Get an issue's body and comment thread. Use this when a PR has linked_issue_numbers, since the original bug report or feature request often has the real 'why'.",
        "input_schema": {
            "type": "object",
            "properties": {"issue_number": {"type": "integer"}},
            "required": ["issue_number"],
        },
    },
    {
        "name": "search_issues",
        "description": "Full-text search across issues and PRs by keyword. Use this when you don't have a specific symbol/number yet, e.g. to find discussions about a general topic.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

SYSTEM_PROMPT = """You are a code archaeologist. Given a question about why a piece of \
code exists or changed, investigate using the available tools the way a senior engineer \
would: start from the symbol or topic, find the commit(s) that touched it, follow the \
pr_number to the PR discussion, and follow linked_issue_numbers to the original issue if \
one exists. Don't stop at the first commit message -- the real rationale is often in the \
PR discussion or linked issue, not the commit message itself.

When you have enough evidence, give a final answer that:
- States the rationale clearly and concisely
- Cites specific commit SHAs, PR numbers, or issue numbers for every claim
- Explicitly says so rather than guessing, if the evidence is inconclusive or you found \
conflicting signals

Do not call tools more than necessary. If get_commit_history returns nothing, try \
search_issues with a looser query before giving up."""


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(question: str, repo: str, dsn: str, api_key: str, verbose: bool = True) -> str:
    tools = ArchaeologyTools(dsn, repo)
    client = anthropic.Anthropic(api_key=api_key)

    messages = [{"role": "user", "content": question}]

    for iteration in range(MAX_TOOL_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            # Model is done reasoning and has produced its final answer
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return final_text

        # Append the assistant's tool-use turn, then run each requested tool
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if verbose:
                print(f"  [tool call] {block.name}({block.input})")

            try:
                result = _dispatch_tool(tools, block.name, block.input)
            except Exception as e:
                result = {"error": str(e)}

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
            })

        messages.append({"role": "user", "content": tool_results})

    return "Reached max tool iterations without a confident answer. The investigation so far is in the conversation log above -- consider narrowing the question."


def _dispatch_tool(tools: ArchaeologyTools, name: str, tool_input: dict):
    if name == "get_commit_history":
        return tools.get_commit_history(tool_input["symbol"])
    elif name == "get_diff":
        return tools.get_diff(tool_input["sha"], tool_input.get("file_path"))
    elif name == "get_pr_discussion":
        return tools.get_pr_discussion(tool_input["pr_number"])
    elif name == "get_issue":
        return tools.get_issue(tool_input["issue_number"])
    elif name == "search_issues":
        return tools.search_issues(tool_input["query"])
    else:
        return {"error": f"Unknown tool: {name}"}


def main():
    parser = argparse.ArgumentParser(description="Ask the Code Archaeology agent a question")
    parser.add_argument("--repo", required=True, help="owner/name, must already be ingested")
    parser.add_argument("--question", required=True)
    parser.add_argument("--quiet", action="store_true", help="suppress tool-call logging")
    args = parser.parse_args()

    dsn = os.environ["DATABASE_URL"]
    api_key = os.environ["ANTHROPIC_API_KEY"]

    answer = run_agent(args.question, args.repo, dsn, api_key, verbose=not args.quiet)
    print("\n--- ANSWER ---\n")
    print(answer)


if __name__ == "__main__":
    main()
