# Phase 1B — Q2 diagnostic-feature prompt (E) validation: negative result

Small validation run per the 4-case design (2 known-hard misses, 1 previously-successful
control, 1 wrong-species case). Raw output:
[results/q2_prompt_e_results.json](results/q2_prompt_e_results.json).

**Result: Prompt E underperformed Prompt C on 3 of 4 cases, including regressing the
control case that C had previously solved perfectly.** This does not confirm the
hypothesis that explicitly prompting for specific diagnostic feature categories (lesion
color, concentric rings, underside structures, etc.) improves retrieval.

| Case | Failure type | C rank | C top1 | E rank | E top1 |
|---|---|--:|---|--:|---|
| D1 cedar apple rust | missing distinctive feature | None | bacterial leaf spot | None | black rot |
| D6 early blight | missing distinctive feature | 12 | septoria leaf spot | 13 | black rot |
| D7 bacterial leaf spot | wrong-species case | **1** | bacterial leaf spot | **2** | black rot |
| D4 powdery mildew | control (C succeeded) | **1** | powdery mildew | **7** | black rot |

D1 stayed unfixed, D6 got marginally worse, D7 dropped from a perfect rank 1, and D4 —
the case Prompt C had solved cleanly — regressed from rank 1 to rank 7. Three of the
four E queries converged on the same wrong top-1 disease ("black rot"), suggesting
something generic/degenerate about how these particular queries embedded, not a
disease-specific improvement.

## Why: the rigid template likely backfired, not the underlying hypothesis

Reading the actual E outputs shows a probable mechanism — **the 10-field checklist format
caused the model to leave most fields blank or terse ("not observed") rather than
producing rich descriptive text**:

- D6's output has literally every field empty except the labels themselves — the model
  echoed the template structure without filling in content.
- D7 and D4 answered "not observed" for 6–7 of the 10 fields, with only one or two fields
  containing any real description.
- D1 got some content (4 of 10 fields answered) but still far sparser than Prompt C's
  free-form bulleted symptom list from the earlier experiment.

Prompt C's format (a shorter, more open "list what you observe" instruction, no fixed
category checklist) produced full, richly descriptive bullets in every case. Prompt E's
rigid 10-category template appears to have made the model default to sparse/negative
answers for categories that didn't obviously apply, actively **removing** descriptive
content rather than adding diagnostic precision — the opposite of the intended effect.

## What this does and doesn't tell us

- **Does not confirm**: that explicitly naming diagnostic feature categories improves
  retrieval. This specific implementation of that idea made things worse.
- **Does not yet tell us**: whether the underlying idea (nudge the model toward
  disease-discriminating features specifically) is wrong, or whether it just needs a
  better implementation — e.g. asking for the same category list but as guidance rather
  than a fill-in-the-blank template, with a larger token budget, or embedded within
  Prompt C's already-working open format rather than replacing it.
- **Confirms again**: Prompt C (structured "Species: unknown" + free bulleted symptom
  list, no rigid field template) remains the best-performing query strategy found so far
  across both experiments.

This is a genuine negative result, not a setup for a positive one — worth deciding
explicitly whether to iterate on the diagnostic-feature idea with a better prompt design,
or accept Prompt C's format as the Q2 answer and move on to Q6 / Milestone 3.
