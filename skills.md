# PAWS Buddy Operational Sub-Skills

This document outlines specialized sub-skills that Gemini agents can execute to maintain and extend the PAWS Buddy codebase.

---

## 1. Sub-Skill: `context-sync`

### Purpose
Scans Git status, diffs, or newly added files and updates [project-map.md](file:///Users/koushiksadhukhan/projects/paws-buddy/project-map.md) and [project-catalog.md](file:///Users/koushiksadhukhan/projects/paws-buddy/project-catalog.md) incrementally without discarding existing custom notes.

### Trigger Pattern
- "Sync project context"
- "Update project catalog after changes"
- Invoked after creating new endpoints, modules, or services.

### Execution Procedure
1. Run `git status -s` and inspect untracked or modified files in `src/`, root scripts, or tests.
2. Check `pyproject.toml` for any new packages added via `uv add`.
3. If new classes, Pydantic models, or endpoints were added:
   - Append class and schema definitions to [project-catalog.md](file:///Users/koushiksadhukhan/projects/paws-buddy/project-catalog.md).
   - Update endpoint tables and dependency lists.
4. Update directory tree and component responsibility table in [project-map.md](file:///Users/koushiksadhukhan/projects/paws-buddy/project-map.md).

---

## 2. Sub-Skill: `generate-code-flow`

### Purpose
Generates text-based or Mermaid sequence and flow diagrams for complex pipelines, request lifecycles, or state transitions to help developers visualize execution paths.

### Trigger Pattern
- "Explain flow of /session/evaluate_frame"
- "Generate sequence diagram for vision pipeline"
- "Show data flow for kinematics hold gate"

### Execution Procedure
1. Identify the target function, endpoint, or class in [pipeline.py](file:///Users/koushiksadhukhan/projects/paws-buddy/pipeline.py) or [app.py](file:///Users/koushiksadhukhan/projects/paws-buddy/app.py).
2. Trace the step-by-step logic, input parameters, transformations, condition gates, and return schemas.
3. Formulate a Mermaid flowchart (`flowchart TD`) or sequence diagram (`sequenceDiagram`) highlighting inputs, decision thresholds, and output states.

---

## 3. Sub-Skill: `amend-context-doc`

### Purpose
Safely appends or updates architectural decisions, ADRs (Architectural Decision Records), or task workflows into [project-architecture.md](file:///Users/koushiksadhukhan/projects/paws-buddy/project-architecture.md) or [GEMINI.md](file:///Users/koushiksadhukhan/projects/paws-buddy/GEMINI.md) without overwriting manual steering directives.

### Trigger Pattern
- "Document architectural decision for ESP32 MQTT integration"
- "Add rule for running tests before commit"
- "Update roadmap in GEMINI.md"

### Execution Procedure
1. Read the current contents of the target context file.
2. Identify the specific target section (e.g. `## Mandatory Coding Rules` or `## Architecture Overview`).
3. Apply targeted non-destructive edits or append new subsections.
4. Ensure all markdown links conform to the `file://` link standard.
