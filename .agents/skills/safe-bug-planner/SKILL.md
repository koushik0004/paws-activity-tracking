---
name: safe-bug-planner
description: Diagnoses errors, analyzes root causes, and outputs a concise fix plan with expected code diffs without modifying any files until explicitly approved. Optimized for planning with Gemini 3.8 Flash (Medium) and applying changes with Gemini 3.8 Flash (Low).
---

# Safe Bug Planner & Cost-Optimized Fix Skill

## Purpose

The `safe-bug-planner` skill provides a strict, two-phase debugging workflow for Gemini CLI / Antigravity (`agy`). It prevents unwanted file edits during investigation by enforcing a read-only planning phase first, and optimizes token/compute costs by separating deep root-cause reasoning from low-effort code application.

## Strict Operational Rules

1. **Zero Automatic File Modifications:**
   * **NEVER** edit, create, delete, or overwrite any workspace files during diagnosis and planning.
   * Only apply file changes after the user explicitly confirms in a follow-up turn (e.g., `"Apply fix"`, `"Approved"`, or `"Go ahead and implement"`).
2. **Brevity & High Signal:**
   * Keep explanations sharp, structured, and free of fluff. Get straight to the root cause, the exact steps to fix it, and the expected code diff.
3. **Model & Effort Cost Optimization Protocol:**
   * **Phase 1 (Planning):** Designed to run while your CLI is set to `Gemini 3.8 Flash` with `/effort medium` for thorough trace analysis.
   * **Phase 2 (Implementation):** Before applying the approved fix, remind the user to switch to `/effort low` so straightforward file writes consume minimal reasoning tokens.

---

## Workflow

### Phase 1: Read-Only Root Cause Analysis & Fix Plan (Default)

Whenever an error, stack trace, or bug report is provided:

1. **Inspect Context (Read-Only):**
   * Read the failing file(s), related schemas/models, and recent git changes (`git diff`, `git status`) without altering anything.
2. **Deliver Concise Output in Chat:**
   Respond using this exact 4-part structure:

   * **1. Root Cause Analysis (2–3 sentences):**
     * Identify the exact file, function/line, and why the failure occurs.
   * **2. Step-by-Step Fix Plan:**
     * Numbered, atomic steps detailing what needs to change (and any required CLI/DB migration commands).
   * **3. Expected Code Changes (`diff` Preview):**
     * Show exact unified `diff` blocks (`-` removed / `+` added) with file paths so the user can review the exact lines before approval.
   * **4. Next Step & Cost-Optimization Prompt:**
     * Explicitly ask for confirmation before touching any files, and include the `/effort low` tip:
       > *"No files have been modified. To apply this fix with minimal token cost, switch to `/effort low` and reply **'Apply fix'**."*

---

### Phase 2: Approved Implementation (Explicit User Trigger Only)

Only execute this phase after the user explicitly approves the plan from Phase 1:

1. **Apply Planned Diffs:**
   * Make only the exact edits agreed upon in Phase 1. Do not perform unrelated refactoring.
2. **Verify:**
   * Run the relevant test, linter, or health check command if applicable, and report a 2-line confirmation of the updated files.