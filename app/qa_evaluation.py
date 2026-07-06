import logging
import pandas as pd
from app.qa import ask, build_qa_prompt, build_qa_prompt_v2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_evaluation():
    df = pd.read_csv("data/questions.csv")
    versions = {
        "v1_strict": build_qa_prompt,
        "v2_strictest": build_qa_prompt_v2,
    }
    all_results = []
    for vname, prompt_fn in versions.items():
        logger.info(f"Evaluating prompt: {vname}")
        answerable_hits = 0
        unanswerable_refusals = 0
        total_answerable = (df["answerable"] == "yes").sum()
        total_unanswerable = (df["answerable"] == "no").sum()

        for _, row in df.iterrows():
            question = row["question"]
            expected = str(row["expected_phrase"]) if pd.notna(row["expected_phrase"]) else ""
            result = ask(question, prompt_fn=prompt_fn)
            answer = result["answer"]
            # Check if answerable answer contains expected phrase
            if row["answerable"] == "yes":
                if expected.lower() in answer.lower():
                    answerable_hits += 1
            else:
                # Unanswerable: refusal is correct
                if result["context_found"] == False and answer.strip().lower() == "i don't know":
                    unanswerable_refusals += 1
                # also consider any 'i don't know' as refusal even if context_found flag is wrong
                elif "i don't know" in answer.lower():
                    unanswerable_refusals += 1

        acc = answerable_hits / total_answerable if total_answerable > 0 else 0
        ref_rate = unanswerable_refusals / total_unanswerable if total_unanswerable > 0 else 0
        logger.info(f"{vname}: answer accuracy={acc:.2%}, refusal rate={ref_rate:.2%}")
        all_results.append({
            "prompt_version": vname,
            "answer_accuracy": acc,
            "refusal_rate": ref_rate
        })

    scores_df = pd.DataFrame(all_results)
    print(scores_df)
    scores_df.to_csv("results/qa_scores.csv", index=False)
    logger.info("Saved QA scores to results/qa_scores.csv")

if __name__ == "__main__":
    run_evaluation()