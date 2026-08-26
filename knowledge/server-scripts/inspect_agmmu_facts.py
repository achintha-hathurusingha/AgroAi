#!/usr/bin/env python3
"""
Phase 1A data inspection: what's actually in agmmu_ft_hf1.json (45,096 entries)?
Checks structure, field completeness, duplicates, and malformed records --
reports findings only, makes no preprocessing decisions.
"""
import json
from collections import Counter
from pathlib import Path

PATH = Path("/home/minura/agrivision-rag/data/agmmu/agmmu_ft_hf1.json")


def main():
    with open(PATH) as f:
        data = json.load(f)

    n = len(data)
    print(f"total entries: {n}\n")

    # --- top-level key structure ---
    top_keys = Counter()
    qa_field_keys = Counter()
    for entry in data:
        top_keys.update(entry.keys())
        qa = entry.get("finetuning qa", {})
        qa_field_keys.update(qa.keys())

    print("=== top-level keys across all entries (key: count out of total) ===")
    for k, c in top_keys.most_common():
        print(f"  {k}: {c}/{n}")

    print("\n=== 'finetuning qa' sub-fields across all entries ===")
    for k, c in qa_field_keys.most_common():
        print(f"  {k}: {c}/{n}")

    # --- malformed / incomplete records ---
    missing_qa = [e for e in data if "finetuning qa" not in e]
    missing_images = [e for e in data if not e.get("images")]
    empty_answers = []
    nonstring_answers = []
    for e in data:
        qa = e.get("finetuning qa", {})
        for field, qa_pair in qa.items():
            ans = qa_pair.get("a", "") if isinstance(qa_pair, dict) else ""
            if not isinstance(ans, str):
                nonstring_answers.append((e.get("faq-id"), field, type(ans).__name__, ans))
                continue
            if not ans or not ans.strip():
                empty_answers.append((e.get("faq-id"), field))

    print(f"\n=== completeness ===")
    print(f"entries missing 'finetuning qa' entirely: {len(missing_qa)}")
    print(f"entries missing/empty 'images': {len(missing_images)}")
    print(f"individual empty answer fields: {len(empty_answers)}")
    if empty_answers[:5]:
        print(f"  sample: {empty_answers[:5]}")
    print(f"individual non-string answer fields: {len(nonstring_answers)}")
    if nonstring_answers[:5]:
        for faqid, field, typ, val in nonstring_answers[:5]:
            print(f"  faq-id={faqid} field={field!r} type={typ} value={val!r}")

    # --- duplicates ---
    # exact duplicate check on the full finetuning-qa content (species+disease+symptom+management)
    def qa_fingerprint(e):
        qa = e.get("finetuning qa", {})
        parts = []
        for field in sorted(qa.keys()):
            pair = qa[field]
            if isinstance(pair, dict):
                ans = pair.get("a", "")
                ans = ans if isinstance(ans, str) else json.dumps(ans, sort_keys=True)
                parts.append(f"{field}:{ans}")
        return "|".join(parts)

    fingerprints = [qa_fingerprint(e) for e in data]
    fp_counts = Counter(fingerprints)
    exact_dupes = {fp: c for fp, c in fp_counts.items() if c > 1}
    print(f"\n=== duplicates ===")
    print(f"unique fact fingerprints: {len(fp_counts)} out of {n} entries")
    print(f"fingerprints appearing more than once: {len(exact_dupes)}")
    dupe_entry_count = sum(exact_dupes.values())
    print(f"entries involved in exact duplication: {dupe_entry_count} ({dupe_entry_count/n*100:.1f}%)")

    # species-level duplicate check (same species+disease, different symptom/mgmt wording)
    def safe_str(v):
        return v.strip().lower() if isinstance(v, str) else str(v)

    def species_disease_key(e):
        qa = e.get("finetuning qa", {})
        species = safe_str(qa.get("species", {}).get("a", "")) if "species" in qa else ""
        disease = safe_str(qa.get("disease/issue identification", {}).get("a", "")) if "disease/issue identification" in qa else ""
        return (species, disease)

    sd_counts = Counter(species_disease_key(e) for e in data)
    print(f"\nunique (species, disease) pairs: {len(sd_counts)}")
    print("top 15 most common (species, disease) pairs:")
    for (sd, c) in sd_counts.most_common(15):
        print(f"  {sd}: {c}")

    # species distribution alone
    species_counts = Counter(species_disease_key(e)[0] for e in data)
    print(f"\nunique species values: {len(species_counts)}")
    print("top 20 species by entry count:")
    for sp, c in species_counts.most_common(20):
        print(f"  {sp!r}: {c}")

    # --- sample entries for eyeballing ---
    print("\n=== 3 sample entries (first, middle, last) ===")
    for idx in [0, n // 2, n - 1]:
        print(f"\n--- entry {idx} (faq-id={data[idx].get('faq-id')}) ---")
        print(json.dumps(data[idx], indent=2)[:800])


if __name__ == "__main__":
    main()
