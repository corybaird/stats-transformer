# Commit Management Workflow: PyPI Python Library Projects

## Objective
Generate git commit messages that follow the project's historical ALL-CAPS verb convention while keeping changes atomic and preserving explicit review gates for open-source package development, testing, and distribution.

## Commit Pattern
Commit messages MUST follow this format:
1.  **ALL-CAPS Verb:** Start with a strong action verb from the dictionary.
2.  **Concise Subject:** A short description of the change (e.g., feature addition, bug fix, version bump).
3.  **No Trailing Punctuation:** Do not end with a period.
4.  **Standard References:** Use lowercase for file extensions (e.g., `.py`, `.toml`, `.md`).

### Action Verb Dictionary
`CREATE`, `UPDATE`, `ADD`, `REMOVE`, `FIX`, `REFACTOR`, `DEPRECATE`, `DOCS`, `RELEASE`, `TEST`.

### Examples
*   `ADD pytest coverage for core module`
*   `FIX version bump in pyproject.toml`
*   `UPDATE dependencies for pydantic v2`
*   `DOCS update api reference for new endpoints`
*   `RELEASE v1.2.0`
*   `REFACTOR simplify loop in data loader`
*   `TEST add edge case tests for string parser`
*   `REMOVE unused legacy helper functions`

## Atomic / Micro Commits
*   **Max Files Per Commit:** Prefer 1-2 files per atomic commit and keep most commits to 3-5 files. If a logical change spans more than 5 files, split it when there is a clean boundary; keep tightly coupled generated or mechanical changes together when splitting would make review harder.
*   **Group Couplings:** Bundle logically related changes (e.g., a new feature `.py` file and its corresponding unit test `test_*.py`).
*   **Differentiate Intent:** Keep source code fixes (`FIX`) separate from CI/CD pipeline changes (`UPDATE`) or documentation (`DOCS`) unless they are intrinsically linked.
*   **Milestone Based:** Commit as you reach logical points in the branch, not just at the very end.
*   **Respect Existing Work:** Do not revert or overwrite unrelated user changes. If unrelated dirty files are present, leave them out of the commit unless the user explicitly includes them.

## Workflow: Commit then PR

When the user requests a commit:

1.  **Status Check:** Run `git status` and `git diff` to review staged/unstaged changes.
2.  **Analyze Scope:** Identify logical atomic units. If more than 5 files, propose splitting into multiple commits.
3.  **Select Verb:** Pick the most accurate verb from the approved list for each commit.
4.  **Draft Summary:** Write the subject line following the pattern.
5.  **Present for Approval:** Show the proposed commit message(s) and file groupings. Wait for user confirmation.
6.  **Commit:** After approval, run `git add` and `git commit -m "<subject>"` for each commit.
7.  **Propose PR:** Once all commits are made, present a PR draft for the user to review and approve before creation.

## Pull Request Workflow

### Before Creating PR
*   Verify branch is clean (`git status` shows no uncommitted changes).
*   Ensure commits are atomic and follow the commit pattern.
*   Ensure the branch is pushed or ask the user before pushing.
*   Confirm the target branch, usually `main`.

### PR Content
The PR should include:
*   **Title:** Clear summary of the feature/fix
*   **Summary:** Bulleted list of key changes
*   **Testing:** Any commands needed to verify (e.g., pytest)
*   **Breaking Changes:** Note any breaking changes if applicable

### PR Gate
*   Do NOT create the PR until the user explicitly approves.
*   Present the PR details to the user for confirmation before running `gh pr create`.

## Release Workflow

When the user requests a release:

1.  **Confirm Scope:** Review `git status`, recent commits, and the diff since the last tag.
2.  **Choose Version:** Recommend a SemVer bump (`PATCH`, `MINOR`, or `MAJOR`) based on user-facing changes.
3.  **Update Metadata:** Bump `version` in `pyproject.toml` and move `CHANGELOG.md` entries from `Unreleased` into a dated release section.
4.  **Verify Package:** Run the relevant tests and build checks before publishing (at minimum `uv run pytest tests` and `uv build` unless the user chooses to skip them).
5.  **Review Artifacts:** Inspect the generated distribution contents for accidental private files, large data, reports, credentials, or local-only agent/runtime state.
6.  **Release Gate:** Do NOT create tags, GitHub releases, or publish to PyPI until the user explicitly approves the final version, changelog, and artifact state.

## Branch Strategy
*   Use feature branches.
*   Branch from `main` for new features/fixes.
*   PRs target `main` unless otherwise specified.
*   Keep branches focused — one feature or fix per branch.
