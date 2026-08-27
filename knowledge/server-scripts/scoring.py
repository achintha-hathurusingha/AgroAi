#!/usr/bin/env python3
"""
Corrected scoring for Phase 2 (and its extension). Fixes two bugs discovered after
Stage F: (1) ground_truth_disease was the raw PlantVillage class label, never
normalized to the corpus's own vocabulary, so 6/16 confident_match classes could never
score "correct" under exact-string match even with perfect retrieval+diagnosis;
(2) Recall@K used full-string equality on the corpus fact's "disease" field, which
undercounts hits when a fact bundles multiple diseases in one comma-separated field.

PV_ALIAS_MAP: PV class -> set of accepted ground-truth strings (PV label + corpus's
own canonical term, where they differ and are confirmed to be the same disease/pest by
different wording). "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)" is deliberately
EXCLUDED -- its coverage-script match ("ascochyta leaf blight") is plausibly a
different pathogen (Isariopsis/Pseudocercospora vitis vs Ascochyta), not just a naming
variant, so forcing that alias risks crediting a scientifically wrong match. Treat that
class as still unresolved pending manual review.
"""

PV_ALIAS_MAP = {
    "apple scab": {"apple scab"},
    "black rot": {"black rot"},
    "cedar apple rust": {"cedar apple rust"},
    "powdery mildew": {"powdery mildew"},
    "cercospora leaf spot gray leaf spot": {"cercospora leaf spot gray leaf spot", "cercospora leaf spot", "gray leaf spot"},
    "bacterial spot": {"bacterial spot", "bacterial leaf spot"},
    "early blight": {"early blight"},
    "leaf scorch": {"leaf scorch"},
    "septoria leaf spot": {"septoria leaf spot"},
    "tomato mosaic virus": {"tomato mosaic virus", "mosaic virus"},
    # deliberately excluded / unresolved:
    # "leaf blight (isariopsis leaf spot)": possibly different pathogen than corpus's
    #   "ascochyta leaf blight" -- not aliased, scored as before (strict).
}


def ground_truth_aliases(gt_pv_label):
    """gt_pv_label: the raw (already-lowercased) ground_truth_disease string from the
    eval set. Returns the set of strings that should count as correct."""
    gt = gt_pv_label.strip().lower()
    return PV_ALIAS_MAP.get(gt, {gt})


def is_correct_aliased(diagnosis, gt_pv_label):
    d = (diagnosis or "").strip().lower()
    return d in ground_truth_aliases(gt_pv_label)


def corpus_disease_terms(disease_field):
    """Split a possibly comma-separated corpus disease field into individual terms."""
    if not disease_field:
        return set()
    return {t.strip().lower() for t in disease_field.split(",") if t.strip()}


def fact_matches_ground_truth(disease_field, gt_pv_label):
    """True if any term in this corpus fact's disease field is an accepted alias of
    the ground truth (handles both the vocabulary-mismatch and multi-value-field bugs
    at once)."""
    return bool(corpus_disease_terms(disease_field) & ground_truth_aliases(gt_pv_label))
