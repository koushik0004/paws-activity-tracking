---
name: gemini-project-init
description: Scans a repository to generate and maintain root context files for Gemini Pro coding agents. Use when initializing a repo or updating project context.
---

# Skill: `gemini-project-init`

## Overview
Scans a repository to generate and maintain root context files (`GEMINI.md`, `project-map.md`, `project-catalog.md`, `project-architecture.md`, `skills.md`) for Gemini Pro coding agents. Use when initializing a repo, updating project context, or adding developer task workflows and code generation instructions.

---

## Trigger Commands / Patterns
- `gemini-project-init`
- `/init`
- "Initialize project context for Gemini"
- "Rescan repo and update project map/catalog"

---

## Execution Workflow

When invoked, Gemini Pro must perform the following 5 actions:

### Action 1: Repository Structure & Dependency Audit
1. Inspect the repository root and subdirectories to locate core packages (`src/`, `lib/`, `tests/`, etc.).
2. Scan dependency definitions (`requirements.txt`, `pyproject.toml`, `package.json`, etc.) to identify active tools, libraries, and frameworks.
3. Locate active entry points (`main.py`, `app.py`, `index.ts`, CLI scripts).

### Action 2: Generate / Update `GEMINI.md`
Create or update `GEMINI.md` at the repository root to establish steering instructions for the Gemini agent:
- Define the **Agent Role & Persona** tailored to the codebase (e.g., Senior AI Architect, Full-Stack Developer).
- Set **Mandatory Coding Rules** (e.g., type hinting, modular architecture, docstring requirements, environment variable handling).
- Reference the core context files (`project-map.md`, `project-catalog.md`, `project-architecture.md`, `skills.md`).
- Document the immediate development roadmap or priorities.

### Action 3: Generate / Update `project-map.md`
Create or update `project-map.md` with:
- An updated ASCII/Markdown directory tree.
- A table listing core entry points, configuration scripts, and key modules.
- Annotation of directory responsibilities (e.g., `src/ingestion/` -> data loading).

### Action 4: Generate / Update `project-catalog.md` & `project-architecture.md`
- **`project-catalog.md`**: Index all major classes, functions, custom agent tools, and external package dependencies.
- **`project-architecture.md`**: Document the high-level system architecture, data flow diagrams (ASCII/Mermaid format), key component behaviors, and evaluation or testing layers.

### Action 5: Generate / Update `skills.md` & Specific Sub-Skills
Create or update `skills.md` containing operational sub-skills that Gemini can execute within this specific repository:
1. **`context-sync`**: Scans diffs/commits and updates `project-map.md` and `project-catalog.md` incrementally whenever new files or modules are created.
2. **`generate-code-flow`**: Generates text-based sequence/flow diagrams for complex functions or pipelines to help developers visualize code execution.
3. **`amend-context-doc`**: Safely appends or updates architectural decisions and task workflows into `project-architecture.md` or `GEMINI.md` upon user request.

---

## Incremental Maintenance Protocol
- If `GEMINI.md` or context files already exist, **do not erase custom steering rules** added by the user.
- Scan for new modules, functions, or dependencies added since the last run.
- Update only the outdated sections in `project-map.md` and `project-catalog.md` to keep diffs clean and context precise.