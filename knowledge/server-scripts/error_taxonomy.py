#!/usr/bin/env python3
"""
Stage A: Error Taxonomy. Data-driven classification using already-collected evidence:
- current pipeline's query_text, retrieved facts (Arm 2 / BGE), diagnosis, confidence, judge output
- oracle retrieval's diagnosis and (from retrieval_decomposition) oracle's retrieval rank

Classification logic per wrong-diagnosis case (disease-in-corpus group):
  1. oracle_retrieval_hit = did the ORACLE query find the correct disease in its top-k?
     (from retrieval_decomposition per_case_ranks["oracle"][case_id] <= 5)
  2. current_retrieval_hit = did the CURRENT (Qwen query) pipeline's actual Arm-2 retrieval
     surface the correct disease in its top-5? (from phase2 arm2.retrieved)

  -> if not oracle_retrieval_hit: RETRIEVAL/CORPUS FAILURE (not findable even with a perfect query)
  -> elif oracle_retrieval_hit and not current_retrieval_hit: QUERY FAILURE (a good query would
     have found it; the actual generated query didn't)
  -> elif current_retrieval_hit: REASONING FAILURE (correct evidence was retrieved, diagnosis
     still wrong)

Overlaid, independent dimensions (can co-occur with any of the above):
  - HALLUCINATION flag: judge marked the diagnosis as not grounded in the evidence
  - CALIBRATION FAILURE flag: confidence >= 70 despite an incorrect diagnosis
"""
import json
import random

with open(r"c:\Users\User\Documents\AgroAi\knowledge\results\phase2\phase2_full_v2_results.jsonl") as f:
    main_rows = {json.loads(l)["case_id"]: json.loads(l) for l in f}

with open(r"c:\Users\User\Documents\AgroAi\knowledge\results\phase2\retrieval_decomposition_results.json") as f:
    retrieval_decomp = json.load(f)
oracle_ranks = retrieval_decomp["per_case_ranks"]["oracle"]

CONFIDENCE_THRESHOLD = 70


def is_correct(diagnosis, gt):
    return (diagnosis or "").strip().lower() == gt.strip().lower()


def retrieval_hit(retrieved_list, gt, k=5):
    gt = gt.strip().lower()
    for r in retrieved_list[:k]:
        if (r.get("disease") or "").strip().lower() == gt:
            return True
    return False


cases = [r for r in main_rows.values() if r["eval_group"] == "primary_confident_match"]
print(f"total confident_match cases: {len(cases)}")

classified = []
for r in cases:
    cid = r["case_id"]
    gt = r["ground_truth_disease"]
    arm2 = r["arms"]["2_rag_bge"]
    diag = arm2.get("diagnosis")
    correct = is_correct(diag, gt)

    oracle_rank = oracle_ranks.get(cid)
    oracle_hit = oracle_rank is not None and oracle_rank <= 5
    current_hit = retrieval_hit(arm2.get("retrieved", []), gt)

    judge = arm2.get("judge", {})
    grounded = judge.get("grounded")
    confidence = arm2.get("confidence")

    if correct:
        category = "CORRECT"
    elif not oracle_hit:
        category = "RETRIEVAL_CORPUS_FAILURE"
    elif oracle_hit and not current_hit:
        category = "QUERY_FAILURE"
    else:
        category = "REASONING_FAILURE"

    hallucination = (grounded is False)
    calibration_failure = (not correct) and (confidence is not None) and (confidence >= CONFIDENCE_THRESHOLD)

    classified.append({
        "case_id": cid, "ground_truth": gt, "diagnosis": diag, "correct": correct,
        "category": category, "oracle_rank": oracle_rank, "current_retrieval_hit": current_hit,
        "confidence": confidence, "grounded": grounded, "hallucination": hallucination,
        "calibration_failure": calibration_failure,
        "query_text": r.get("query_text"), "retrieved_top1_disease": (arm2.get("retrieved") or [{}])[0].get("disease"),
        "reasoning": arm2.get("reasoning"),
    })

from collections import Counter
cat_counts = Counter(c["category"] for c in classified)
print("\n=== Primary category counts (all 240 confident_match cases) ===")
for cat, n in cat_counts.most_common():
    print(f"  {cat:28s} {n:3d}  ({n/len(classified)*100:.1f}%)")

n_hallucination = sum(1 for c in classified if c["hallucination"])
n_calib_fail = sum(1 for c in classified if c["calibration_failure"])
n_wrong = sum(1 for c in classified if not c["correct"])
print(f"\nHallucination (not grounded, among wrong cases): {n_hallucination}/{n_wrong} = {n_hallucination/n_wrong*100:.1f}%")
print(f"Calibration failure (conf>={CONFIDENCE_THRESHOLD} despite wrong, among wrong cases): {n_calib_fail}/{n_wrong} = {n_calib_fail/n_wrong*100:.1f}%")

# stratified sample of ~70 cases for qualitative review: mix of categories
random.seed(42)
sample = []
for cat in ["RETRIEVAL_CORPUS_FAILURE", "QUERY_FAILURE", "REASONING_FAILURE", "CORRECT"]:
    pool = [c for c in classified if c["category"] == cat]
    take = min(len(pool), 18 if cat != "CORRECT" else 10)
    sample += random.sample(pool, take)
print(f"\nstratified sample size: {len(sample)}")

with open(r"c:\Users\User\Documents\AgroAi\knowledge\results\phase2\error_taxonomy_full.json", "w") as f:
    json.dump(classified, f, indent=2)
with open(r"c:\Users\User\Documents\AgroAi\knowledge\results\phase2\error_taxonomy_sample.json", "w") as f:
    json.dump(sample, f, indent=2)

print("\n=== representative examples per category ===")
for cat in ["RETRIEVAL_CORPUS_FAILURE", "QUERY_FAILURE", "REASONING_FAILURE"]:
    examples = [c for c in sample if c["category"] == cat][:3]
    print(f"\n--- {cat} ---")
    for e in examples:
        print(f"  case={e['case_id'][:50]:50s} gt={e['ground_truth']!r:25s} diag={e['diagnosis']!r:25s} conf={e['confidence']}")
