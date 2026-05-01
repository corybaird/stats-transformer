# Agent Review Structure and Workflow

This directory contains automated audits and refactoring validations comparing legacy academic code against the `stats-transformer` library. 

When conducting a new review for a paper (e.g., `apep_XXXX`), you must follow these structured parameters for both the **content** and the **shipping process**.

---

## 1. Directory and Content Structure

Create a dedicated folder for the paper: `src/examples/agent_reviews/apep_XXXX/`
Within this folder, you must produce exactly two artifacts:

### A. Review Markdown (`review.md`)
The markdown review must be formatted cleanly and include the following sections:
- **Title and Source**: State the paper title and the source code path.
- **🚩 Critical Implementation Flaws**: Document 2-3 major antipatterns found in the legacy code (e.g., manual matrix inversions, loop-based regressions, hardcoded dictionaries). For each flaw, explicitly lay out the **Issue** and the **Stats-Transformer Solution**.
- **📉 Impact on Outcomes**: Explicitly detail how the legacy approach impacts the final empirical results or execution reliability. Does it lead to biased standard errors? Overstated significance? Runtime inefficiencies? If original data is unavailable, state that the tests were run on representative or simulated architectures.
- **🛠️ Refactoring Snippet Comparison**: Show a side-by-side code snippet of the legacy approach vs. the `stats-transformer` approach.
- **✅ Conclusion**: A brief summary of why the `stats-transformer` architecture is statistically and programmatically superior for this specific paper.

### B. Replication Script (`replicate_real.py`)
The Python script must programmatically prove the claims made in the review. The script must define a class `ApepXXXXRealReplication` containing the following required methods:
- `fetch_real_data(self)`: Fetches the original data (via API like FRED, or loading local CSVs). If original data cannot be found or loaded, this method should cleanly generate representative simulated data to prove the structural differences.
- `calculate_original(self, df)`: Implements the exact (or highly approximated) legacy logic found in the original paper (e.g., manual `np.linalg.inv`, manual loops, non-robust standard errors).
- `calculate_statstransformer(self, df)`: Implements the clean, robust solution using the `stats-transformer` library's estimators (like `RobustOLSModel`) or vectorization tools (like `FeatureEngineer`).
- `compare_methods(self, df)`: Executes both calculation methods and prints a clear, quantified comparison (e.g., difference in standard errors, difference in coefficients, execution time comparison) to the console.
- `run(self)`: The singular entry point that orchestrates fetching and comparing.

---

## 2. Technical Execution Rules

The replication file must strictly adhere to the project's global rules:
- The script must be executable cleanly via `uv run python -m src.examples.agent_reviews.apep_XXXX.replicate_real`.
- **Never use docstrings**.
- **All function and method arguments must be on a single line**.
- **Never use type checking**.
- **Never create a `main()` function** (use `if __name__ == "__main__":` to instantiate the class and call `.run()`).

---

## 3. GitHub Shipping Workflow

Reviews must never land directly on `main`. Follow this end-to-end flow for every review:

### Phase 1 — Branch Creation
**Cardinal rule: never edit on main.** If on `main`, switch to a feature branch:
```bash
git switch -c docs/review-apep-XXXX
```
Convention: Use `docs/review-` or `feat/review-` prefix.

### Phase 2 — Logical Commits
Split the work into small, logically-grouped commits. Do not dump all files in one mega-commit.
- **Commit 1**: Create directory and `review.md`.
- **Commit 2**: Implement `replicate_real.py`.
- **Commit 3**: Update any central tracking or indices.

**Commit Message Format:** `VERB <concise subject>` (ALL-CAPS verb)
- `DOCS create review for apep_XXXX`
- `ADD replication script for apep_XXXX validation`

### Phase 3 — Push and Pull Request
Push the branch and open a PR using the GitHub CLI:
```bash
git push -u origin <branch-name>
gh pr create --base main --head <branch-name> --title "DOCS: Review and Replication for apep_XXXX" --body "Summary of the review findings and replication results."
```

**The user merges the PR manually in the GitHub UI. The agent never merges.**

---

## 4. Operating Principles
- **Multiple small commits > one giant commit.**
- **One PR = one review.**
- **Match the existing repo's style.**
- **The user merges. Always.**
