# -*- coding: utf-8 -*-
"""Run the full BanglaPlagPair-1K experiment suite end to end, in order.

    python run_all_experiments.py            # everything
    python run_all_experiments.py --from 4   # resume from step 4
    python run_all_experiments.py --only 2 3 # just those steps

Every step is idempotent: re-running overwrites its own outputs and never
touches ``BanglaPlagPair-1K_source_links_final.xlsx``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

from pipeline_common import banner


STEPS = {
    1: ("Build canonical base dataset", "step1_build_base_dataset"),
    2: ("Traditional metrics - RAW dataset", "step2_score_raw_dataset"),
    3: ("Polish dataset + traditional metrics", "step3_polish_and_score"),
    4: ("BanglaBERT evaluation (2 models x raw/polished)", "step4_run_banglabert_evaluation"),
    5: ("Preprocessing component ablation", "step5_preprocessing_component_ablation"),
    6: ("Consolidate results + integrity audit", "step6_consolidate_results"),
    7: ("Sentence-embedding model evaluation", "step7_sentence_embedding_evaluation"),
    8: ("Free Hugging Face LLM evaluation", "step8_free_llm_evaluation"),
    9: ("Top-3 prompt-engineering evaluation", "step9_prompt_engineering_evaluation"),
    10: ("Paid LLM evaluation", "step10_paid_llm_evaluation"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BanglaPlagPair-1K experiment suite.")
    parser.add_argument("--from", dest="start", type=int, default=1, help="First step to run")
    parser.add_argument("--to", dest="stop", type=int, default=max(STEPS), help="Last step to run")
    parser.add_argument("--only", type=int, nargs="+", default=None, help="Run only these steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = sorted(args.only) if args.only else [n for n in sorted(STEPS) if args.start <= n <= args.stop]

    timings: list[tuple[int, str, float]] = []
    saved_argv = sys.argv
    for number in selected:
        title, runner = STEPS[number]
        banner(f"RUNNING STEP {number}: {title}")
        started = time.time()
        # Some steps parse sys.argv themselves; hide this script's flags from them
        # so they fall back to their documented defaults.
        sys.argv = [f"step{number}"]
        try:
            importlib.import_module(runner).main()
        finally:
            sys.argv = saved_argv
        elapsed = time.time() - started
        timings.append((number, title, elapsed))
        print(f"\n  Step {number} finished in {elapsed:.1f}s")

    banner("SUITE COMPLETE")
    for number, title, elapsed in timings:
        print(f"  step {number}  {elapsed:8.1f}s  {title}")


if __name__ == "__main__":
    main()
