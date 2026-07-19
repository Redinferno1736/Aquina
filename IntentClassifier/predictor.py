from sklearn.preprocessing import LabelEncoder
import torch

class AquinaIntentPredictor:
    """
    Self-contained intent prediction wrapper for Aquina.

    Usage after unpickling:
        predictor = pickle.load(open("aquina_intent_pipeline.pkl", "rb"))
        intent, confidence = predictor.predict("open chrome please")
    """

    def __init__(self, model_dir: str, label_encoder: LabelEncoder, max_length: int = 64):
        self.model_dir = model_dir
        self.label_encoder = label_encoder
        self.max_length = max_length

        # These are NOT pickled (see __getstate__ below) — they are
        # rebuilt lazily on first use in the new process/environment.
        self._model = None
        self._tokenizer = None
        self._device = None

    def _lazy_load(self):
        """Load the model and tokenizer from disk if not already loaded."""
        if self._model is None or self._tokenizer is None:
            from transformers import AutoTokenizer, DistilBertForSequenceClassification

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = DistilBertForSequenceClassification.from_pretrained(self.model_dir)
            self._model.to(self._device)
            self._model.eval()

    def predict(self, text: str):
        """
        Predict the intent for a single text string.
        Returns: (predicted_intent: str, confidence: float)
        """
        self._lazy_load()

        inputs = self._tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            confidence, pred_id = torch.max(probs, dim=-1)

        intent = self.label_encoder.inverse_transform([pred_id.item()])[0]
        return intent, confidence.item()

    def predict_batch(self, texts: list):
        """
        Predict intents for a list of text strings.
        Returns: list of (intent: str, confidence: float) tuples
        """
        self._lazy_load()

        inputs = self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            confidences, pred_ids = torch.max(probs, dim=-1)

        intents = self.label_encoder.inverse_transform(pred_ids.cpu().numpy())
        return list(zip(intents, confidences.cpu().numpy().tolist()))

    # --------------------------------------------------------------------
    # Custom pickling behavior: only serialize the lightweight metadata
    # (model_dir path + label_encoder). The actual torch model/tokenizer
    # are excluded from the pickle and reloaded lazily after unpickling.
    # --------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        state["_tokenizer"] = None
        state["_device"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)