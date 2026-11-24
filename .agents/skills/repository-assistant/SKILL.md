---
name: repository-assistant
description: Comprehensive workspace assistant for code investigation, architecture documentation, targeted QA, and step-by-step issue resolution with optional human-in-the-loop (HITL) approval.
---

# Repository Assistant Skill

## Purpose
The `repository-assistant` skill provides a flexible, multi-mode workflow for answering questions about the repository, generating standalone analysis reports, providing targeted answers without touching code, or fixing issues interactively with human feedback.

---

## Operating Modes & Intent Recognition

Determine the execution mode based on the user's prompt intent:

### Mode 1: Direct Inquiry / Targeted Q&A (No File Changes)
* **Trigger Criteria:** User asks simple questions, logic explanations, file location checks, or explicitly requests concise/to-the-point answers with *no file edits*.
* **Behavior:**
  1. Inspect the codebase using read-only operations.
  2. Synthesize a concise, direct answer in the chat window.
  3. **Do not** create or edit any files.

---

### Mode 2: Markdown Analysis & Report Generation
* **Trigger Criteria:** User requests comprehensive deep dives, module breakdowns, security/performance audits, or explicitly asks for output saved in a `.md` file (e.g., `ANALYSIS.md`, `ARCHITECTURE_REVIEW.md`).
* **Behavior:**
  1. Scan relevant source files, classes, and architectural flows.
  2. Structure the analysis logically (Overview, Architecture, Bottlenecks/Issues, Recommended Fixes).
  3. Write or update the requested `.md` document in the workspace.
  4. Provide a brief chat summary referencing the generated markdown file.

---

### Mode 3: Automated Issue Diagnosis & Fix
* **Trigger Criteria:** User reports a bug, exception, stack trace, or asks to locate and repair a specific issue autonomously.
* **Behavior:**
  1. Locate the bug source by inspecting error traces and relevant code modules.
  2. Formulate a minimal, safe code fix adhering to existing project standards.
  3. Apply the changes to the necessary code files.
  4. Summarize the root cause and verified changes in the chat.

---

### Mode 4: Human-in-the-Loop (HITL) Interactive Troubleshooting
* **Trigger Criteria:** User specifies "step-by-step", "ask before editing", "human in the loop", or asks for interactive confirmation during complex refactoring.
* **Behavior:**
  1. Analyze the codebase and diagnose the problem.
  2. Present the findings and proposed fix plan to the user in the chat *first*.
  3. Ask for explicit user confirmation before applying file changes.
  4. Apply changes only after receiving user approval.

---

## Execution Guidelines

1. **Context Awareness:** Always check project steering files (`GEMINI.md`, `project-map.md`, `project-catalog.md`) if available to maintain consistent conventions.
2. **Minimal Footprint:** For direct queries or HITL proposals, refrain from making unrequested file changes.
3. **Traceability:** When generating analysis reports (`.md`), link to specific source filenames and line references where appropriate.