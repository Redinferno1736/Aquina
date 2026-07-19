import pickle
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from predictor import AquinaIntentPredictor

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------
PIPELINE_PATH = "models/aquina_intent_pipeline.pkl"
REALITY_CSV = "aquina_reality_check.csv"


def main():
    # --------------------------------------------------------------------
    # Step 1: Load pipeline
    # --------------------------------------------------------------------
    print(f"Loading pipeline from {PIPELINE_PATH}...")

    with open(PIPELINE_PATH, "rb") as f:
        predictor = pickle.load(f)

    print("Pipeline loaded successfully.")

    # --------------------------------------------------------------------
    # Step 2: Load reality-check CSV
    # --------------------------------------------------------------------
    print(f"Loading {REALITY_CSV}...")

    df = pd.read_csv(REALITY_CSV)
    df = df[["text", "intent"]].copy()

    df.dropna(subset=["text", "intent"], inplace=True)
    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df.reset_index(drop=True, inplace=True)

    print(f"Loaded {len(df)} examples.\n")

    # --------------------------------------------------------------------
    # Step 3: Predict
    # --------------------------------------------------------------------
    texts = df["text"].tolist()
    true_intents = df["intent"].tolist()

    results = predictor.predict_batch(texts)

    predicted_intents = [intent for intent, conf in results]
    confidences = [conf for intent, conf in results]

    # --------------------------------------------------------------------
    # Step 4: Print per-example results
    # --------------------------------------------------------------------
    print("=" * 100)
    print("PER-EXAMPLE RESULTS")
    print("=" * 100)

    correct_count = 0

    for text, true_intent, pred_intent, conf in zip(
        texts,
        true_intents,
        predicted_intents,
        confidences,
    ):
        correct = true_intent == pred_intent
        correct_count += int(correct)

        marker = "✅" if correct else "❌"

        print(
            f'{marker} "{text}"\n'
            f'    true: {true_intent:<20} '
            f'pred: {pred_intent:<20} '
            f'confidence: {conf:.3f}'
        )

    # --------------------------------------------------------------------
    # Step 5: Summary
    # --------------------------------------------------------------------
    accuracy = accuracy_score(true_intents, predicted_intents)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(f"Correct: {correct_count}/{len(texts)}")
    print(f"Reality-check accuracy: {accuracy:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            true_intents,
            predicted_intents,
            zero_division=0,
        )
    )

    # --------------------------------------------------------------------
    # Step 6: Low-confidence predictions
    # --------------------------------------------------------------------
    LOW_CONFIDENCE_THRESHOLD = 0.70

    low_conf = [
        (t, ti, pi, c)
        for t, ti, pi, c in zip(
            texts,
            true_intents,
            predicted_intents,
            confidences,
        )
        if c < LOW_CONFIDENCE_THRESHOLD
    ]

    if low_conf:
        print("=" * 100)
        print(
            f"LOW CONFIDENCE PREDICTIONS (< {LOW_CONFIDENCE_THRESHOLD})"
        )
        print("=" * 100)

        for text, true_intent, pred_intent, conf in low_conf:
            print(
                f'"{text}" -> '
                f'pred={pred_intent} '
                f'(true={true_intent}) '
                f'conf={conf:.3f}'
            )
    else:
        print(
            f"No predictions below {LOW_CONFIDENCE_THRESHOLD:.2f} confidence."
        )


if __name__ == "__main__":
    main()