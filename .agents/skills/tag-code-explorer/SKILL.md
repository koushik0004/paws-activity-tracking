---
name: tag-code-explorer
description: Rapidly explores and explains the target goal, architecture, execution flow, code changes, and annotated snippets with citations of a specific Git branch or tag by prioritizing README files, testing Jupyter notebooks (.ipynb), and core source diffs. Outputs either concise chat summaries or blog-ready Markdown walkthroughs with Mermaid diagrams.
---

# Tag & Branch Code Explorer Skill

## Purpose

The `tag-code-explorer` skill helps developers quickly understand large incoming codebases after checking out a new Git branch or release tag. It synthesizes the product/engineering target, end-to-end execution flow, exact code changes, and cited code snippets by prioritizing human-written documentation (`README.md`) and runnable test notebooks (`.ipynb`) alongside source files. It also structures comprehensive `.md` reports so they can easily be adapted into technical blog posts or social media content.

## Discovery & Analysis Workflow

Whenever this skill is invoked, execute the following discovery pipeline in order:

1. **Identify Git Context & Code Changes:**
   * Check the current Git state (`git branch --show-current`, `git describe --tags --exact-match` or `git log -n 1 --oneline`) to confirm the active tag, branch, or commit.
   * Identify what changed in this tag/branch by inspecting recent commits or diffs against the previous tag/commit (`git tag --sort=-creatordate`, `git diff --stat <prev_tag>..HEAD`, or `git status`).

2. **Prioritize Anchor Files (High-Signal Context):**
   * **Step A (`README.md`):** Locate and read the relevant `README.md` (root and tag/week-specific subdirectories) to extract the stated target ("What We're Building"), learning/engineering objectives, architecture overview, and prerequisites.
   * **Step B (Testing `.ipynb` Notebooks):** Search for Jupyter notebooks (`*.ipynb`, especially integration, demo, or verification notebooks). Parse the markdown cells and code cells in execution order—along with their stdout/stderr execution results—to understand how components are wired together, tested end-to-end, and any known caveats (e.g., schema migrations or container rebuilds).

3. **Trace Core Source Implementation & Citations:**
   * Follow the imports, factories, and function calls seen in the `.ipynb` test cells back into the core source directory (e.g., `src/services/`, `src/repositories/`, `src/db/`, `dags/`).
   * Record exact file paths, function/class names, and line numbers (or notebook cell numbers) so every code snippet includes a verifiable citation.

## Output Modes & Intent Recognition

Adapt your output format based on the user's request:

### Mode 1: Concise / Precise Answer (Chat Only)

* **Trigger:** User asks a specific question about the tag/branch ("What does this tag build?", "How is X tested in the notebook?", "Give me a quick overview") without requesting a file.
* **Format:**
  * **Tag/Branch Target:** 1–2 sentences summarizing what this version builds and why.
  * **Execution Flow:** A brief step-by-step bullet list derived from the `.ipynb` and `README.md`.
  * **Key Cited Snippets:** 1–3 short, annotated code snippets with file/cell citations.
  * **Constraint:** Do **not** create or modify any workspace files.

### Mode 2: Blog-Ready Technical & Product Walkthrough Saved to `.md` File

* **Trigger:** User provides a target filename (e.g., `WEEK2_WALKTHROUGH.md`, `TAG_V2_BLOG_NOTES.md`) or asks to save the walkthrough/findings to a markdown document.
* **Format:** Write a clean, engaging, publication-ready Markdown file structured so the developer can both master the codebase and effortlessly adapt it into a technical blog post or social media thread (LinkedIn/X/Substack):

  1. **Executive Summary & Product Target ("What We Built & Why"):**
     * Hook/problem statement, target goal of the specific Git tag/branch, business/system value, and key performance/success metrics.
  2. **System Architecture & End-to-End Flow (Mermaid Diagram):**
     * A clear, valid `mermaid` flowchart (`flowchart TD` or `sequenceDiagram`) illustrating how data and control move across components (e.g., API -> Client -> Parser -> Orchestrator -> Database/Workflow).
     * Step-by-step narrative walkthrough of the pipeline.
  3. **What Changed in This Tag (Code & Infrastructure Diff):**
     * Summary of newly added services, repositories, schemas, DAGs, and infrastructure/container updates compared to the prior stage.
  4. **Core Code Walkthrough with Minimal Docs & Citations:**
     * Key implementation snippets (5–20 lines each) covering factories, core services, orchestrators, and repository operations.
     * Every snippet **must** include:
       * Minimal, crystal-clear documentation explaining *what* it does and *why* it matters.
       * Exact **Citation** (e.g., `Source: src/services/metadata_fetcher.py (lines 42–65)` or `Notebook: notebooks/week2/week2_arxiv_integration.ipynb [Cell 10]`).
  5. **Verification & Real-World Gotchas (`.ipynb` Test Insights):**
     * Breakdown of how the test notebook validates each component, including real operational caveats discovered during testing (e.g., rate limiting, schema updates, container dependencies).
  6. **Blog & Social Media Takeaways (Ready-to-Share Angles):**
     * **Key Engineering Takeaways:** 3–5 bullet points highlighting architectural lessons learned.
     * **Social Post Draft / Hook:** A concise, punchy summary draft ready for LinkedIn/X summarizing what was built in this milestone.

* **Action:** Save the file to the user-specified path and provide a brief confirmation summary in the chat.

## Execution Guidelines

* **Read Notebooks Cleanly:** When inspecting `.ipynb` files, parse cell `source` content (code and markdown) and summarized `outputs`; ignore noisy base64 image blobs or raw execution metadata.
* **Valid Mermaid Syntax:** Always enclose node labels containing special characters (parentheses, slashes, colons) in double quotes (e.g., `A["ArxivClient (3s Rate Limit)"] --> B["PDFParserService (Docling)"]`) so diagrams render without syntax errors.
* **Citation Discipline:** Never paste unattributed code blocks. Always attach the exact file path and line range or notebook cell reference.
* **Strict File Discipline:** Never write `.md` files unless the user explicitly asks to save the findings or provides a target filename.