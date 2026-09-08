import Foundation

/// The built-in prompt presets and their resolution — `whiz/ai.py`'s prompt
/// layer, transcribed verbatim. The prompt text IS the product here: every
/// section heading, dedup instruction, owner/effort rule and the Essentials
/// augmentation shape the artifact users actually read, so fidelity to the
/// Python strings is fidelity to the behavior. Tests pin the load-bearing
/// markers (section names, tokens, the Essentials section) rather than every
/// word, so upstream prompt tuning doesn't churn the suite.

// MARK: - Built-in prompts

enum AnalysisPrompts {

    static let summary = """
    You are an expert meeting assistant. Below is a transcript of a meeting (with speaker labels and timestamps). Produce a concise summary:
    - A 2-3 sentence overview of what the meeting was about.
    - Key topics discussed (bullet points).
    - Any decisions made.

    Transcript:
    {transcript}

    Summary:
    """

    static let actions = """
    You are an expert meeting assistant. Below is a transcript of a meeting (with speaker labels and timestamps). Extract action items:
    - One bullet per action item.
    - Format: '- [owner] action (by deadline if mentioned)'.
    - Owner = the speaker who committed to it; use '?' if unclear.
    - Only include concrete tasks, not general discussion points.
    If there are no action items, say 'No action items found.'

    Transcript:
    {transcript}

    Action items:
    """

    static let summaryAndActions = """
    You are an expert meeting assistant. Below is a transcript of a meeting (with speaker labels and timestamps). Produce:

    ## Summary
    A 2-3 sentence overview, then key topics as bullet points, then any decisions made.

    ## Action items
    One bullet per action item: '- [owner] action (by deadline if mentioned)'. Owner = the speaker who committed to it; '?' if unclear. Only concrete tasks. If none, write 'No action items found.'

    Transcript:
    {transcript}
    """

    static let plan = """
    You are an expert technical lead. Below is a transcript of a discussion (with speaker labels and timestamps) that is about building a feature, fixing a bug, or carrying out a technical task. Turn it into a concrete, actionable implementation plan.

    CRITICAL — distinguish the WORK from the MEETING. The transcript records what people SAID and DID during a call ('Vika shares screen', 'Vika opens DevTools', 'Vika explains the difference', 'wrap up the session'). Those are events of the discussion itself, NOT implementation steps. Your Steps, Risks, and Open questions must describe the engineering work to be DONE AFTER the call, not narrate what happened IN the call.
    - A step like 'Share screen and navigate to the page' is a meeting event — do not list it. A step like 'Add a Clearinghouse entity to the data model' is implementation work — list that.
    - A risk like 'Vika is not sure what a clearinghouse is' is a fact about the meeting, not a build risk. A risk like 'The payer API may return unsettled claims that need filtering' is a build risk.
    - If the call is PRIMARILY an explanation, walkthrough, or knowledge-transfer session (one person showing/describing an existing system to another) rather than an active decision-making discussion about what to build, do NOT force a build plan. Instead say so under Overview and produce session notes: key facts learned, entities/fields/workflows explained, decisions made (if any), and open questions — not Steps/Risks/Acceptance criteria that describe the conversation.

    Produce exactly these sections, in this order, using Markdown headings:

    ## Overview
    2-4 sentences: what is being built/changed and why — OR, if this is a walkthrough/explanation session, state that explicitly and say what was explained.

    ## Goal
    The single concrete outcome that 'done' looks like.

    ## Proposed approach
    A short narrative of how the work will be done — the key design choice and its rationale.

    ## Steps
    A numbered list of implementation tasks to be done AFTER the call, not a narration of what happened during the call. Each step MUST have:
    - **Step N.** <title> — one line of what to do (the engineering work).
    - **Owner:** the named speaker who raised or owns this task, pulled from the speaker label on the transcript line where it was discussed. Use the actual speaker name/label — NOT a generic role like 'Dev'. If multiple speakers contributed, name the one who owns the task. Use '?' ONLY if no speaker is identifiable for that step.
    - **Effort:** S / M / L, each followed by a one-line justification for that estimate (e.g. 'M — new endpoint + 2 tests'). Never give a bare size without the reason.
    List every concrete task inferred from the transcript; if a task is only implied, mark it '(inferred)'. Keep the list in a sensible execution order.

    ## Risks
    Bullet list of risks and unknowns of the BUILD, each with a short mitigation — not observations about the participants or the meeting.

    ## Open questions
    Bullet list of unresolved questions that need a decision or more info. DEDUPLICATE: merge near-identical questions into one before output — never list the same question twice (same meaning, different wording counts as a duplicate). If none, write 'None.'

    ## Acceptance criteria
    A checklist ('- [ ] ...') of the conditions the finished work must meet to be considered done. Pull these from the transcript; infer reasonable ones if the discussion was thin.

    Transcript:
    {transcript}
    """

    static let classify = """
    You are a fast content classifier. Read the transcript below and decide which of these three categories it belongs to:
    - MEETING — people discussing a past/ongoing topic, a standup, a review, a decision meeting, an interview, or general conversation.
    - PLAN — people actively deciding how to build, implement, fix, or change a specific feature, bug, product, or technical task, where the output should be an actionable implementation plan rather than meeting notes.
    - WALKTHROUGH — primarily one person explaining, showing, or describing an existing system, codebase, domain, or workflow to others (knowledge transfer / a tour), even if a future task is mentioned. The output should be session notes (what was explained, key facts, open questions), not a build plan.

    Reply with EXACTLY ONE token: MEETING, PLAN, or WALKTHROUGH. No other text, no punctuation.

    Transcript:
    {transcript}

    Classification:
    """

    static let walkthrough = """
    You are an expert technical scribe. Below is a transcript of a call (with speaker labels and timestamps) that is primarily a walkthrough, explanation, or knowledge-transfer session — one person showing or describing an existing system, codebase, domain, or workflow to another. Produce session notes that capture what was explained, NOT an implementation plan and NOT a narration of what happened during the call.

    Produce exactly these sections, in this order, using Markdown headings:

    ## Overview
    2-4 sentences: what system/topic was walked through and who was explaining to whom.

    ## Key facts learned
    A dense bullet list of the substantive facts conveyed — entities, fields, schemas, API responses, workflows, relationships, terminology, and how things work. Prefix with timestamp and speaker when useful. This is the reference a future reader (or a later AI) should treat as the durable takeaway of the session.

    ## Decisions
    Bullet list of any decisions made during the session. If none, write 'None.'

    ## Open questions
    Bullet list of unresolved questions raised or implied. DEDUPLICATE: merge near-identical questions into one. If none, write 'None.'

    ## Suggested next steps
    Bullet list of the implementation work that this session implies or motivates (if any) — the work to be done AFTER the call, not events of the call itself. If the session was purely explanatory with no follow-up work, write 'None.'

    Transcript:
    {transcript}
    """

    // MARK: - Essentials (always-on augmentation)

    /// Shared analyst posture, prepended to every Essentials augmentation.
    /// Applies to the WHOLE analysis: maximum effort, frame/transcript
    /// reconciliation with confidence tags, visual-timeline reasoning, and
    /// the 'FUN:' attention to the human texture of the call.
    static let analystPosture = """
    Be exceptionally thorough and attentive. Reason step-by-step at maximum effort before producing any section; do not rush to the first plausible answer. When on-screen frames are provided, you have TWO sources of truth — the spoken transcript AND the visible screen. Actively reconcile them: cross-check names, labels, field values, button text, error messages, and UI state shown on screen against what was said, and surface discrepancies (mark them 'SCREEN vs TRANSCRIPT:'). Treat the frames as authoritative for anything visible (schema, code, config, URLs) and the transcript as authoritative for intent and discussion; use both. CRITICAL: be conservative with screen-derived claims. NEVER assert a contradiction between screen and transcript unless BOTH the on-screen text AND the transcript text are legibly readable. If either is blurry, partial, occluded, or you are inferring what it says, do NOT claim a discrepancy — note it as an observation instead. Every 'SCREEN vs TRANSCRIPT:' item MUST end with a confidence tag: [HIGH] (both clearly readable and the mismatch is unambiguous), [MEDIUM] (readable but interpretation involved), or [LOW] (one or both sides are unclear/partial). If you cannot confidently read both sides, omit the item entirely rather than guess. When multiple frames are provided for a chunk, treat them as a VISUAL TIMELINE, not independent screenshots. A single topic, decision, or unit of sense often spans several consecutive frames — reason across the sequence, not frame-by-frame in isolation. Transitions between adjacent frames (UI state changes, new elements appearing, values changing, panels opening/closing) are as meaningful as any single frame's content. Anchor your reconciliation to the window of frames + transcript lines together, not to individual frame-transcript pairs. Pay attention to the HUMAN texture of the call, not just the dry facts. Speakers coin slang, invented words, in-jokes, meme-y language, and absurd little moments — these are real signal about how the team thinks and feels, not noise to filter out. Actively notice and quote them (in the original language, then a one-line gloss in English if needed), and mark each one 'FUN:' so they surface instead of being skipped.
    """

    private static let essentialsTask =
        "After producing the analysis above, ALSO produce a `## Essentials` "
        + "section: a dense, exhaustive bullet list of EVERY meaningful point — facts, "
        + "decisions, requirements, constraints, names, numbers, UI/UX details, "
        + "workflows, open questions, and rejected alternatives. One bullet per point, "
        + "concise; prefix with timestamp and speaker when useful "
        + "(e.g. '- [00:12:03] Vadim: ...'). Mark open questions 'OPEN:', rejected "
        + "alternatives 'REJECTED:', inferred points '(inferred)', and any coined "
        + "slang / in-jokes / absurd or funny moments 'FUN:' (quote the original "
        + "wording, then gloss in English if it's not English). With frames, also "
        + "capture visible on-screen text/schema. This section is for feeding to a "
        + "later AI analysis as concentrated context."

    /// Rides on the {task} label in MAP/SYNTH for built-in prompts.
    static let essentialsTaskSuffix = analystPosture + essentialsTask

    /// Appended to single-call and custom paths where there is no {task} slot,
    /// inserted right after the {transcript} placeholder so the model reads
    /// the transcript first, then the posture + Essentials instruction.
    static let essentialsInstruction = "\n\n---\n" + analystPosture + essentialsTask

    /// Appended to the custom reduce prompt so the synth merges the per-chunk
    /// Essentials sections into one.
    static let essentialsReduceInstruction = """

    The partial answers above may each contain a `## Essentials` section. Merge ALL of those Essentials bullets into one consolidated, deduplicated `## Essentials` section at the end of your final answer, preserving timestamps and speaker prefixes.
    """

    // MARK: - Map-reduce prompts

    static let mapPrompt = """
    You are analyzing chunk {k} of {n} of a longer recorded transcript. Your job: {task}

    {context_block}Analyze the transcript chunk below. Keep speaker labels and timestamps. Build on the running context above when present — refer back to speakers, decisions, entities, and open threads already established; do not re-derive them from scratch. Do not invent anything not supported by this chunk or the running context. Produce a partial result for THIS chunk that fits coherently with what came before; a later step will merge all chunks.

    Transcript chunk ({k}/{n}):
    {transcript}
    """

    static let contextBlock = """
    Running context from prior chunks (their partial analyses, in order):
    {context}

    Continue from this context.

    """

    static let synthPrompt = """
    You are combining {n} partial analyses of a long recorded transcript into one final answer. Your job: {task}

    Below are the {n} partial analyses, one per contiguous chunk, in time order. They were produced with rolling context, so later partials already refer back to earlier ones. Merge them into a single coherent answer: remove duplicates, reconcile conflicts, keep the chronological order, and preserve specific speaker/time references. Deduplicate the Open questions section especially: merge near-identical questions (same meaning, different wording) into one. Produce the final answer in the exact format the task expects.

    Partial analyses:
    {partials}
    """

    static let customReducePrompt = """
    Below are {n} partial answers, one per contiguous chunk of a long transcript, in time order. They were produced with rolling context, so later answers already refer back to earlier ones. Merge them into one final answer: remove duplicates, reconcile conflicts, and preserve specific speaker/time references. Do not add anything not supported by the partials.

    Partial answers:
    {partials}
    """

    // MARK: - Resolution

    /// The Swift shape of ai.py's identity-keyed prompt globals: a built-in
    /// case or a custom template carried inline.
    enum AIPrompt: Equatable, Sendable {
        case summary
        case actions
        case summaryAndActions
        case plan
        case custom(String)

        var template: String {
            switch self {
            case .summary: return AnalysisPrompts.summary
            case .actions: return AnalysisPrompts.actions
            case .summaryAndActions: return AnalysisPrompts.summaryAndActions
            case .plan: return AnalysisPrompts.plan
            case .custom(let text): return text
            }
        }

        var isBuiltIn: Bool {
            if case .custom = self { return false }
            return true
        }

        /// ai.py:_task_label — the human-readable description that fills
        /// {task} in MAP_PROMPT / SYNTH_PROMPT so built-ins reproduce the
        /// same output structure the single-call path would.
        var taskLabel: String {
            switch self {
            case .summary:
                return "produce a concise meeting summary — a 2-3 sentence overview, key topics as "
                    + "bullet points, and any decisions made"
            case .actions:
                return "extract action items — one bullet per item as '- [owner] action "
                    + "(by deadline if mentioned)'; owner = the speaker who committed to it, '?' "
                    + "if unclear; only concrete tasks"
            case .summaryAndActions:
                return "produce a meeting summary (overview + key topics + decisions) followed by "
                    + "action items (one bullet per item as '- [owner] action (by deadline)')"
            case .plan:
                return "produce a structured implementation plan with sections: Overview, Goal, "
                    + "Proposed approach, Steps (each with Owner = the named speaker who raised it "
                    + "+ Effort S/M/L with a one-line justification), Risks, Open questions "
                    + "(deduplicated), Acceptance criteria"
            case .custom:
                return "answer the user's question about the transcript (the user's prompt "
                    + "is applied to each chunk verbatim)"
            }
        }
    }

    /// ai.py:_augment_prompt_essentials — insert the always-on Essentials
    /// instruction right after the {transcript} placeholder.
    static func augmentPromptEssentials(_ template: String) -> String {
        if template.contains("{transcript}") {
            return template.replacingOccurrences(
                of: "{transcript}", with: "{transcript}" + essentialsInstruction)
        }
        return template + essentialsInstruction
    }
}