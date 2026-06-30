Architecture: Code Archaeology
1. Ingestion layer
A pipeline pulls three correlated data streams from the GitHub API: commit history (diffs + messages), PR metadata (title, description, review comments, linked issues), and issue threads. Store these in a Postgres table keyed by SHA/PR number with foreign keys linking commits → PRs → issues. This relational backbone is what lets you later answer "why" instead of just "what."
2. Symbol-to-history linking
For each commit, run a diff parser (tree-sitter works well across languages) to extract which functions/classes/symbols were touched. Build a mapping table: symbol → [list of commits that touched it]. This is the index that lets a query like "why does parse_config() look like this" jump straight to relevant history instead of searching all commits.
3. Embedding + retrieval layer
Embed PR descriptions, review comments, and issue bodies (not raw diffs — diffs embed poorly) into a vector store. Use hybrid retrieval: vector similarity for semantic matches plus the symbol-table lookup for exact code-path matches. This hybrid approach is itself a good talking point in interviews — pure RAG would miss precise structural relationships.
4. Agent loop (the core differentiator)
Instead of stuffing retrieved context into one prompt, give the LLM tools:

get_commit_history(symbol) — returns commits touching that symbol
get_pr_discussion(pr_id) — returns review thread
get_diff(sha) — returns the actual code change
search_issues(query) — semantic search over issues

The agent decides which tools to call based on the question, chains calls (e.g., find commit → find linked PR → find linked issue → read discussion), and only then synthesizes an answer. This mirrors how a human engineer would actually investigate, and demonstrates real agentic reasoning rather than single-shot RAG.
5. Citation & grounding layer
Every claim the agent makes gets tagged with the source (commit SHA, PR number) it came from. This is critical for the eval step and also makes demos far more convincing — you can click through to verify.
6. Evaluation harness
Hand-pick 20-30 well-documented historical changes in a popular repo where the "why" is verifiable from a linked issue or PR description. Write a small scorer (even LLM-as-judge against ground truth, with manual spot-checks) measuring whether the agent's stated rationale matches the actual documented reason. This is the piece most portfolio projects skip, and it's the piece that signals rigor to a hiring manager.
7. Interface
A simple FastAPI backend + lightweight frontend (could even be a CLI for the MVP) where you type a function name or question and watch the agent's tool calls stream in real time — that visibility into the reasoning process is what makes for a great live demo.
