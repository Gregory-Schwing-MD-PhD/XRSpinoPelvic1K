---
title: Femoral Head Annotation
emoji: 🦴
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Femoral head annotation

Two clicks per lateral radiograph: left femoral head centre, then right.

## Why this exists

The hip point in XRSpinoPelvic1K is learned from **synthetic DRRs**, where the ground
truth is a 3-D sphere fitted to the femoral head and projected. BUU — the only large
public set of real lateral films — annotates L1–L5 and S1 and **nothing pelvic**, so
there is currently no way to measure whether that synthetic hip point survives contact
with a real radiograph. Without it, PI cannot be validated at all.

These annotations are a **reference set, not training data**. Test split only.

## Rules that make the reference worth having

**No automatic proposal is ever shown.** Not the model's prediction, not the classical
circle fit. Displaying a starting point would anchor the annotator to it, and agreement
would then measure suggestibility rather than anatomy.

**Both heads, separately.** The bicoxofemoral point is defined as the midpoint of the two
femoral head centres (Legaye & Duval-Beaupère 1998). Marking both lets the midpoint be
derived, lets their separation flag an oblique film, and lets you mark one when the other
is genuinely invisible.

**Skip freely.** A forced guess on an unreadable film is worse than no annotation — this
set exists to be the reference, so a fabricated point becomes a fabricated error in
whatever it validates.

**Two annotators per film**, with the mean taken on agreement (≤2% of image width) and
disagreements escalated to an adjudicator. Same claim/slot/TTL model as the segmentation
review Space, so it should feel familiar.

## Setup

Space secrets:

    HF_TOKEN       write token for the ledger dataset
    ANNOT_REPO     <org>/xrsp-femhead-annot
    IMAGE_REPO     <org>/xrsp-femhead-images
    ADJUDICATORS   comma/space-separated HF usernames

Seed the ledger and upload the films:

    python annot/seed_cases.py --buu data/BUU-LSPINE --splits data/buu_splits.json \
        --annot-repo <org>/xrsp-femhead-annot --image-repo <org>/xrsp-femhead-images \
        --n 120 --apply

Annotators paste their own HF token; the Space verifies it with `whoami` and uses the
returned username as identity. It caches only `sha256(token) -> username`, never the token.
Films are streamed **through** the Space, so annotators need no read access to any dataset.
