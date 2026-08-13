"""Create/update the femoral-head annotation Space and set its configuration.

    HF_TOKEN=... python annot/deploy.py [--space owner/name] [--adjudicators a,b]

Idempotent: re-running redeploys the code and leaves the ledger untouched, so this is
also the "push a fix" command.

NOTE ON HOSTING. HuggingFace now requires a PRO subscription to create a Docker Space on
free cpu-basic hardware ("Static Spaces are free for everyone, but hosting Gradio and
Docker Spaces on free cpu-basic requires a PRO subscription"). Spaces created before that
change still run. This tool needs a server -- it holds the claim ledger, streams the
films so readers never get access to the image repo, and enforces the two-reader rule --
so a static Space cannot host it.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    # The live Space, and the only one that can be redeployed: creating a NEW Docker Space
    # now needs PRO, so this default is not a preference, it is the one that exists. The
    # ledger and image DATASET repos below are separate and keep their own names.
    ap.add_argument("--space", default="gregoryschwingmdphd/spinesurg-ct-annotator")
    # The LIVE ledger, and this default is load-bearing: deploy.py writes ANNOT_REPO on
    # every run, so a stale default here silently points the Space back at whatever it
    # used to read. That happened -- a routine redeploy reverted the Space to the finished
    # circle-tool ledger and readers got "nothing left to annotate" again. Whenever the
    # live ledger changes, change it HERE, not only in the Space settings.
    ap.add_argument("--annot-repo",
                    default="gregoryschwingmdphd/xrsp-femhead-asp-pilot")
    ap.add_argument("--image-repo", default="gregoryschwingmdphd/xrsp-femhead-images")
    ap.add_argument("--adjudicators", default="gregoryschwingmdphd")
    ap.add_argument("--lock-tool", default="arc",
                    help="pin the annotation primitive ('' to allow switching). One "
                         "accidental press of the Tool button drops circle-tool reads "
                         "into a landmark-tool ledger.")
    ap.add_argument("--claim-ttl", type=int, default=4 * 3600,
                    help="seconds a claimed film stays claimed. 3 days suited a 2000-film "
                         "queue; on a 100-film pilot a reader who closes the tab takes "
                         "5%% of the pool out until the weekend.")
    ap.add_argument("--private", action="store_true",
                    help="private Space; readers then need to be added as collaborators")
    a = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or ""
    if not token:
        cached = Path.home() / ".cache/huggingface/token"
        token = cached.read_text().strip() if cached.exists() else ""
    if not token:
        print("set HF_TOKEN (a write token)", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    # Check FIRST, create only if absent. create_repo(exist_ok=True) still goes through
    # the create endpoint, and that endpoint is what enforces the PRO requirement -- so
    # on a free account it 402s even for a Space that already exists and runs fine.
    try:
        api.space_info(a.space)
        exists = True
    except Exception:                                          # noqa: BLE001
        exists = False

    if exists:
        print(f"  {a.space} exists — redeploying into it")
    else:
        try:
            api.create_repo(a.space, repo_type="space", space_sdk="docker",
                            private=a.private, exist_ok=True)
        except Exception as exc:                               # noqa: BLE001
            if "402" in str(exc):
                print("\n  HuggingFace refused to CREATE the Space: Docker Spaces on\n"
                      "  free cpu-basic now need PRO (https://huggingface.co/pro).\n"
                      "  Spaces that already exist keep working — pass\n"
                      "  --space owner/existing-name to redeploy into one of those.\n",
                      file=sys.stderr)
            raise

    api.upload_folder(
        folder_path=str(HERE), repo_id=a.space, repo_type="space",
        # tests, the figure generator and its ostk dependency have no business in the
        # image; example/reference PNGs DO ship, they are what the readers look at
        ignore_patterns=["__pycache__/*", "*.pyc", "test_annot.py", "make_reference.py",
                         "deploy.py"],
        commit_message="deploy femoral-head annotator")

    # LOCK_TOOL and a short CLAIM_TTL are pilot settings, set here rather than by hand so
    # a redeploy cannot quietly drop them -- which is exactly how ANNOT_REPO reverted the
    # Space to the finished ledger once already.
    for k, v in {"ANNOT_REPO": a.annot_repo, "IMAGE_REPO": a.image_repo,
                 "ADJUDICATORS": a.adjudicators,
                 "LOCK_TOOL": a.lock_tool,
                 "CLAIM_TTL_SECONDS": str(a.claim_ttl)}.items():
        api.add_space_variable(a.space, k, v)
    # The ledger write token. A secret, not a variable: variables are visible in the UI.
    api.add_space_secret(a.space, "HF_TOKEN", token)

    # The app lives on <owner>-<name>.hf.space, NOT huggingface.co/spaces/<owner>/<name>
    # (that is the repo page) and NOT hf.space/<owner>/<name> (which 404s).
    owner, _, name = a.space.partition("/")
    host = f"https://{owner}-{name}".replace("_", "-").lower() + ".hf.space"
    print(f"\n  repo   : https://huggingface.co/spaces/{a.space}")
    print(f"  READERS: {host}")
    print(f"  board  : {host}/board")
    if api.space_info(a.space).private:
        print("\n  This Space is PRIVATE — readers cannot open it until they are added\n"
              "  under Settings -> Collaborators. Making it public would let any\n"
              "  HuggingFace account sign in and pull the films.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
