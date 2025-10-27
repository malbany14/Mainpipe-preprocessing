import pandas as pd
from pipeline import Pipeline
from initial_cleaning import NullCleaningStep
from initial_cleaning import UTF8EncodingStep
from initial_cleaning import LanugageCleaningStep
from initial_cleaning import CaseNormalisationStep
import json
from initial_cleaning import HtmlCleaningStep
from initial_cleaning import SpecialCharacterCleaningStep
from deduplication import ExactDeDuplicationStep
from deduplication import FuzzyDeduplicationStep
from validators import GeneralValidator
import datetime
from pii_and_toxicity import PiiRemovalStep
from pii_and_toxicity import ToxicRemovalStep
from initial_cleaning import QualityFilteringSTep
import numpy as np
import os
from tokenise import TokenizationStep

CHUNK_SIZE = 300000
INPUT_FILE = "../../data/raw/mainpipe_data_v1.jsonl"
OUTPUT_FILE = "../../data/cleaned/cleaned_csv_test.JSONL"

def main():
    # Pipeline cleaning steps
    nullCleaningStep = NullCleaningStep("Clean nulls",GeneralValidator())
    htmlCleaningStep = HtmlCleaningStep("Clean Html", GeneralValidator())
    utf8cCleaningStep = UTF8EncodingStep("Encode to utf8", GeneralValidator())
    specialCharacterCleaningStep = SpecialCharacterCleaningStep("Clean special characters", GeneralValidator())
    exactDeDuplicationStep = ExactDeDuplicationStep("Exact deduplication", GeneralValidator()) # note this is document level
    fuzzyDeduplicationStep = FuzzyDeduplicationStep("Fuzzy deduplification", GeneralValidator()) # this is paragraph leve;
    languageCleaningStep = LanugageCleaningStep("Language cleaning", GeneralValidator())
    caseNormalisationStep = CaseNormalisationStep("Lowercase step", GeneralValidator())
    piiRemovalStep = PiiRemovalStep("PII removal step", GeneralValidator())
    toxicityRemovalStep = ToxicRemovalStep("Toxicity removal step", GeneralValidator()) # NOT USED IN FULL DATASET
    qualityFilteringStep = QualityFilteringSTep("Quality filtering", GeneralValidator())

    # Tokeniser step
    tokeniserStep = [TokenizationStep("gpt2", GeneralValidator())]

    
    # full pipeline steps
    steps = [nullCleaningStep, utf8cCleaningStep, htmlCleaningStep, specialCharacterCleaningStep,
             qualityFilteringStep, languageCleaningStep, exactDeDuplicationStep, fuzzyDeduplicationStep,
             piiRemovalStep, toxicityRemovalStep, caseNormalisationStep]
    
    # testing: last steps
    # steps = [htmlCleaningStep, utf8cCleaningStep, specialCharacterCleaningStep]
    step_reports = []
    

    pipeline = Pipeline(steps, tokeniserStep)
    # ammended for batch loading
    batch = []
    first_chunk = True

    # run pipeline implemented with batching
    MAX_ROWS = 5000 # for testing purposes
    row_count = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if row_count >= MAX_ROWS:
                break  # stop after 5000 rows
            try:
                batch.append(json.loads(line))
                row_count += 1
            except json.JSONDecodeError:
                continue

            if len(batch) >= CHUNK_SIZE:
                df = pd.DataFrame(batch)
                df_clean, df_tokenised = pipeline.run(df)

                # save cleaned text data to jsonl
                with open(OUTPUT_FILE, "a", encoding="utf-8") as out_file:
                    for record in df_clean.to_dict(orient="records"):
                        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                # save tokenised df to numpy
                token_file = OUTPUT_FILE.replace(".JSONL", "_tokens.npy")
                token_arrays = np.array(df_tokenised["token_ids"].tolist(), dtype=object)

                if not os.path.exists(token_file):
                    np.save(token_file, token_arrays)
                else:
                    existing = np.load(token_file, allow_pickle=True)
                    combined = np.concatenate((existing, token_arrays), axis=0)
                    np.save(token_file, combined)

                batch = []

        # Process remaining lines in batch
        if batch:
            df = pd.DataFrame(batch)
            df_clean, df_tokenised = pipeline.run(df)

            # Save cleaned text
            with open(OUTPUT_FILE, "a", encoding="utf-8") as out_file:
                for record in df_clean.to_dict(orient="records"):
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Save tokenized arrays
            token_file = OUTPUT_FILE.replace(".JSONL", "_tokens.npy")
            token_arrays = np.array(df_tokenised["token_ids"].tolist(), dtype=object)

            if not os.path.exists(token_file):
                np.save(token_file, token_arrays)
            else:
                existing = np.load(token_file, allow_pickle=True)
                combined = np.concatenate((existing, token_arrays), axis=0)
                np.save(token_file, combined)

    print(f"All batches processed {OUTPUT_FILE}")
    for step in pipeline.steps:
        metrics = {
            'step_name': step.name,
            'runtime_sec': step.stats.get('runtime_sec', None),
            'removed_rows': len(step.removed_rows) if hasattr(step, 'removed_rows') else None,
            'validator_stats': step.validator.stats
        }
        step_reports.append(metrics)

    # Export repots
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = "../../reports"
    os.makedirs(report_dir, exist_ok=True)  # Ensure the directory exists

    report_file = os.path.join(report_dir, f"pipeline_report_{timestamp}.csv")

    pd.DataFrame(step_reports).to_csv(report_file, index=False)


if __name__ == "__main__":
    main()