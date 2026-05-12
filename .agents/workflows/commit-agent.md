# Commit Management Workflow

## Objective
Generate git commit messages that strictly adhere to the project's historical ALL-CAPS verb convention while ensuring atomicity.

## Commit Pattern
Commit messages MUST follow this format:
1.  **ALL-CAPS Verb:** Start with a strong action verb from the dictionary.
2.  **Concise Subject:** A short description of the change.
3.  **No Trailing Punctuation:** Do not end with a period.
4.  **Standard References:** Use lowercase for file extensions (e.g., `.py`, `.yaml`).

### Action Verb Dictionary
`CREATE`, `UPDATE`, `ADD`, `REMOVE`, `FIX`, `REFACTOR`, `DEPRECATE`, `DOCS`.

### Examples
*   `ADD transformer integration to pipeline`
*   `FIX date formatting in text loader`
*   `UPDATE params.yaml with new model paths`
*   `DOCS update readme with transformer details`

## Atomic / Micro Commits
*   **Max Files Per Commit:** You MUST strictly limit commits to a maximum of 3-5 files. If a logical change spans more than 5 files, you must break it down into smaller, micro commits.
*   **Group Couplings:** Bundle logically related changes (e.g., a new `.py` script and its corresponding `.yaml` entry).
*   **Differentiate Intent:** Keep code fixes (`FIX`) separate from documentation updates (`DOCS`) unless they are intrinsically linked.
*   **Milestone Based:** Commit as you reach logical points in the branch, not just at the very end.

## Workflow Execution
1.  **Status Check:** Review staged files with `git status` and `git diff`.
2.  **Select Verb:** Pick the most accurate verb from the approved list.
3.  **Draft Summary:** Write the subject line following the pattern.
4.  **Draft Body (Optional):** Add a bulleted list for complex changes, separated by a blank line.
5.  **Propose:** Present the message for approval.
6.  **Commit:** Run `git commit -m "<subject>"` after approval.
