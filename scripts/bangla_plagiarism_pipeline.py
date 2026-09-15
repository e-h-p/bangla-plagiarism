# -*- coding: utf-8 -*-
"""Research-grade Bangla preprocessing and similarity evaluation pipeline.

The script reads the controlled Bangla plagiarism-pair workbook used in this
project, applies a conservative preprocessing pipeline, and writes before-vs-
after similarity results with the same metric families used in the draft code.

Default input layout:
    row 5: headers
    rows 6-105: 100 cases
    col A: test case id
    col B: original text
    col C-G: morphological, word-order, synonymy, lexical, mixed variants
Default preprocessing profile:
    lexical_safe = Unicode + punctuation + lowercasing + digit normalization
                   + lexical normalization, without stopword removal/stemming.

Stopword removal and stemming are implemented, but should be reported as an
ablation for this dataset because they can over-normalize lexical/synonymy
variants when the lexical resource is still incomplete.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import Levenshtein
except ImportError:  # pragma: no cover - fallback for clean environments.
    Levenshtein = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "textcase.xlsx"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "preprocessed_outputs"
DEFAULT_LEXICAL_MAP = SCRIPT_DIR / "lexical_normalization.tsv"
DEFAULT_STOPWORDS = SCRIPT_DIR / "bangla_stopwords.txt"

VARIATION_COLUMNS = [
    ("Morphological", 2),
    ("Order", 3),
    ("Synonymy", 4),
    ("Lexical", 5),
    ("Mixed", 6),
]

METRICS = ["Cosine", "Jaccard", "LCS", "Sequence", "EditDist", "WordOverlap"]

INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u2060]")
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u0980-\u09FF]+")

DIGIT_TRANSLATION = str.maketrans(
    {
        "০": "0",
        "১": "1",
        "২": "2",
        "৩": "3",
        "৪": "4",
        "৫": "5",
        "৬": "6",
        "৭": "7",
        "৮": "8",
        "৯": "9",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
)

BENGALI_VOWEL_SIGNS = set("ািীুূৃেৈোৌ")
BENGALI_INDEPENDENT_VOWELS = set("অআইঈউঊঋএঐওঔ")


def collapse_spaces(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def strip_punctuation_and_symbols(text: str) -> str:
    """Replace punctuation/symbols with spaces so neighboring words do not fuse."""

    chars: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category[0] in {"P", "S"} or char == "_":
            chars.append(" ")
        else:
            chars.append(char)
    return "".join(chars)


def basic_normalize(
    text: object,
    *,
    unicode_form: str = "NFC",
    lowercase: bool = True,
    normalize_digits: bool = True,
    remove_punctuation: bool = True,
) -> str:
    """Normalize text without applying stopwords, lexical mapping, or stemming."""

    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    normalized = unicodedata.normalize(unicode_form, str(text))
    normalized = INVISIBLE_RE.sub("", normalized)
    if normalize_digits:
        normalized = normalized.translate(DIGIT_TRANSLATION)
    if lowercase:
        normalized = normalized.lower()
    if remove_punctuation:
        normalized = strip_punctuation_and_symbols(normalized)
    return collapse_spaces(normalized)


def tokenize(text: object) -> list[str]:
    normalized = basic_normalize(text)
    return TOKEN_RE.findall(normalized)


def is_bangla_token(token: str) -> bool:
    return bool(re.fullmatch(r"[\u0980-\u09FF]+", token))


def load_stopwords(path: Path) -> set[str]:
    stopwords: set[str] = set()
    if not path.exists():
        return stopwords

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        stopwords.update(tokenize(line))
    return stopwords


@dataclass(frozen=True)
class LexicalEntry:
    category: str
    canonical: tuple[str, ...]
    variant: tuple[str, ...]
    confidence: float
    note: str = ""


class LexicalNormalizer:
    """Token-level lexical mapper with confidence filtering and n-gram support."""

    def __init__(self, entries: Iterable[LexicalEntry], min_confidence: float = 0.85):
        self.mapping: dict[tuple[str, ...], tuple[str, ...]] = {}
        self.max_variant_len = 1

        for entry in entries:
            if entry.confidence < min_confidence:
                continue
            if not entry.variant or not entry.canonical:
                continue
            if entry.variant == entry.canonical:
                continue
            self.mapping[entry.variant] = entry.canonical
            self.max_variant_len = max(self.max_variant_len, len(entry.variant))

    @classmethod
    def from_tsv(cls, path: Path, min_confidence: float = 0.85) -> "LexicalNormalizer":
        entries: list[LexicalEntry] = []
        if not path.exists():
            return cls(entries, min_confidence=min_confidence)

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                try:
                    confidence = float(row.get("confidence", "0").strip())
                except ValueError:
                    confidence = 0.0

                entries.append(
                    LexicalEntry(
                        category=row.get("category", "").strip(),
                        canonical=tuple(tokenize(row.get("canonical", ""))),
                        variant=tuple(tokenize(row.get("variant", ""))),
                        confidence=confidence,
                        note=row.get("note", "").strip(),
                    )
                )

        return cls(entries, min_confidence=min_confidence)

    def normalize(self, tokens: Sequence[str]) -> list[str]:
        if not self.mapping or not tokens:
            return list(tokens)

        output: list[str] = []
        i = 0
        while i < len(tokens):
            matched = False
            upper = min(self.max_variant_len, len(tokens) - i)
            for size in range(upper, 0, -1):
                candidate = tuple(tokens[i : i + size])
                canonical = self.mapping.get(candidate)
                if canonical is None:
                    continue
                output.extend(canonical)
                i += size
                matched = True
                break
            if not matched:
                output.append(tokens[i])
                i += 1
        return output


class ConservativeBanglaStemmer:
    """Conservative rule-based morphological normalizer for Bangla.

    This is deliberately a stemmer, not a full lemmatizer. The rules target
    frequent case, classifier, plural, and verb-inflection variation in the
    controlled plagiarism dataset while avoiding broad semantic rewriting.
    """

    def __init__(self) -> None:
        self.protected_words = {
            "না",
            "নয়",
            "নয়",
            "নেই",
            "কখনো",
            "কখনও",
            "আমি",
            "আমরা",
            "তুমি",
            "তোমরা",
            "আপনি",
            "আপনারা",
            "সে",
            "তিনি",
            "তারা",
            "এই",
            "ওই",
            "সেই",
            "আজ",
            "গতকাল",
            "আগামীকাল",
            "মধ্যে",
            "সুন্দর",
        }

        self.irregular_verbs = {
            "গেছি": "যা",
            "গেছে": "যা",
            "গিয়েছি": "যা",
            "গিয়েছে": "যা",
            "গিয়েছি": "যা",
            "গিয়েছে": "যা",
            "গেলাম": "যা",
            "গেলেন": "যা",
            "গেল": "যা",
            "এসেছি": "আস",
            "এসেছে": "আস",
            "এসেছিলেন": "আস",
            "এসেছিলাম": "আস",
            "আসব": "আস",
            "আসবে": "আস",
            "হয়েছে": "হও",
            "হয়েছে": "হও",
            "হয়েছিল": "হও",
            "হয়েছিল": "হও",
            "হলো": "হও",
            "হল": "হও",
        }

        self.long_noun_suffixes = sorted(
            {
                "গুলোকে",
                "গুলিকে",
                "গুলোতে",
                "গুলিতে",
                "গুলোর",
                "গুলির",
                "গুলো",
                "গুলি",
                "সমূহকে",
                "সমূহে",
                "সমূহের",
                "সমূহ",
                "দেরকে",
                "দের",
                "টির",
                "টিকে",
                "টিতে",
                "টি",
                "টার",
                "টাকে",
                "টাতে",
                "টা",
                "েরা",
                "রা",
                "দের",
                "কে",
                "তে",
            },
            key=len,
            reverse=True,
        )

        self.verb_suffixes = sorted(
            {
                "েছিলাম",
                "েছিলেন",
                "েছিলে",
                "েছিল",
                "ছিলাম",
                "ছিলেন",
                "ছিলে",
                "ছিল",
                "েছেন",
                "েছিস",
                "েছো",
                "েছি",
                "েছে",
                "ছেন",
                "ছিস",
                "ছো",
                "ছি",
                "ছে",
                "বেন",
                "বিনা",
                "বো",
                "বি",
                "বে",
                "ব",
                "লাম",
                "লেন",
                "লি",
                "লে",
                "তাম",
                "তেন",
                "তে",
            },
            key=len,
            reverse=True,
        )

        self.progressive_suffixes = sorted(
            {
                "চ্ছিলাম",
                "চ্ছিলেন",
                "চ্ছিল",
                "চ্ছেন",
                "চ্ছিস",
                "চ্ছো",
                "চ্ছি",
                "চ্ছে",
            },
            key=len,
            reverse=True,
        )

        self.eya_suffixes = sorted(
            {
                "েয়েছিলাম",
                "েয়েছিলেন",
                "েয়েছিল",
                "েয়েছেন",
                "েয়েছি",
                "েয়েছে",
                "েয়েছিলাম",
                "েয়েছিলেন",
                "েয়েছিল",
                "েয়েছেন",
                "েয়েছি",
                "েয়েছে",
            },
            key=len,
            reverse=True,
        )

        self.iya_suffixes = sorted(
            {
                "িয়েছিলাম",
                "িয়েছিলেন",
                "িয়েছিল",
                "িয়েছেন",
                "িয়েছি",
                "িয়েছে",
                "িয়েছিলাম",
                "িয়েছিলেন",
                "িয়েছিল",
                "িয়েছেন",
                "িয়েছি",
                "িয়েছে",
            },
            key=len,
            reverse=True,
        )

    @staticmethod
    def _strip_suffix(word: str, suffixes: Sequence[str], min_stem_len: int) -> str:
        for suffix in suffixes:
            if word.endswith(suffix):
                candidate = word[: -len(suffix)]
                if len(candidate) >= min_stem_len:
                    return candidate
        return word

    def normalize_verb(self, word: str) -> str:
        original = word

        if word in self.irregular_verbs:
            return self.irregular_verbs[word]

        for suffix in self.eya_suffixes:
            if word.endswith(suffix):
                candidate = word[: -len(suffix)]
                if candidate:
                    return candidate + "া"

        for suffix in self.iya_suffixes:
            if word.endswith(suffix):
                candidate = word[: -len(suffix)]
                if candidate:
                    return candidate + "ে"

        for suffix in self.progressive_suffixes:
            if word.endswith(suffix):
                candidate = word[: -len(suffix)]
                if len(candidate) >= 2:
                    return candidate

        word = self._strip_suffix(word, self.verb_suffixes, min_stem_len=2)

        if word.endswith("ে") and len(word) > 3:
            word = word[:-1]

        if len(word) < 2:
            return original
        return word

    def normalize_noun(self, word: str) -> str:
        original = word
        word = self._strip_suffix(word, self.long_noun_suffixes, min_stem_len=3)

        if word != original:
            return word

        if word.endswith("ের") and len(word) > 3:
            return word[:-2]

        if word.endswith("র") and len(word) > 3:
            candidate = word[:-1]
            final_char = candidate[-1]
            if final_char in BENGALI_VOWEL_SIGNS or final_char in BENGALI_INDEPENDENT_VOWELS:
                return candidate

        return word

    def stem(self, token: str) -> str:
        if not token or token in self.protected_words or not is_bangla_token(token):
            return token

        original = token
        token = self.normalize_verb(token)
        token = self.normalize_noun(token)

        if len(token) < 2:
            return original
        return token


@dataclass(frozen=True)
class PreprocessConfig:
    unicode_form: str = "NFC"
    lowercase: bool = True
    normalize_digits: bool = True
    remove_punctuation: bool = True
    apply_lexical_mapping: bool = True
    remove_stopwords: bool = True
    apply_stemming: bool = True
    min_lexical_confidence: float = 0.85


PROFILE_DEFAULTS: dict[str, dict[str, bool]] = {
    "basic": {
        "apply_lexical_mapping": False,
        "remove_stopwords": False,
        "apply_stemming": False,
    },
    "lexical_safe": {
        "apply_lexical_mapping": True,
        "remove_stopwords": False,
        "apply_stemming": False,
    },
    "full": {
        "apply_lexical_mapping": True,
        "remove_stopwords": True,
        "apply_stemming": True,
    },
}


class BanglaPreprocessor:
    def __init__(
        self,
        config: PreprocessConfig,
        *,
        lexical_map_path: Path = DEFAULT_LEXICAL_MAP,
        stopword_path: Path = DEFAULT_STOPWORDS,
    ) -> None:
        self.config = config
        self.stopwords = load_stopwords(stopword_path)
        self.lexical_normalizer = LexicalNormalizer.from_tsv(
            lexical_map_path, min_confidence=config.min_lexical_confidence
        )
        self.stemmer = ConservativeBanglaStemmer()

    def preprocess_tokens(self, text: object) -> list[str]:
        normalized = basic_normalize(
            text,
            unicode_form=self.config.unicode_form,
            lowercase=self.config.lowercase,
            normalize_digits=self.config.normalize_digits,
            remove_punctuation=self.config.remove_punctuation,
        )
        tokens = TOKEN_RE.findall(normalized)

        if self.config.apply_lexical_mapping:
            tokens = self.lexical_normalizer.normalize(tokens)
        if self.config.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        if self.config.apply_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]

        return tokens

    def preprocess_text(self, text: object) -> str:
        return " ".join(self.preprocess_tokens(text))


def safe_cell(df: pd.DataFrame, row_idx: int, col_idx: int) -> str:
    if row_idx < 0 or col_idx < 0 or row_idx >= df.shape[0] or col_idx >= df.shape[1]:
        return ""
    value = df.iat[row_idx, col_idx]
    return "" if pd.isna(value) else str(value)


def read_controlled_pairs(input_path: Path, start_row: int, end_row: int | None) -> pd.DataFrame:
    raw = pd.read_excel(input_path, header=None, engine="openpyxl")
    if end_row is None:
        end_row = raw.shape[0]

    rows: list[dict[str, str]] = []
    for excel_row in range(start_row, end_row + 1):
        idx = excel_row - 1
        case_id = safe_cell(raw, idx, 0).strip()
        if not case_id:
            case_id = str(excel_row)

        record: dict[str, str] = {
            "Test Case No": case_id.replace(".0", "") if case_id.endswith(".0") else case_id,
            "Original Text": safe_cell(raw, idx, 1),
        }

        for variation, col_idx in VARIATION_COLUMNS:
            record[f"{variation} Text"] = safe_cell(raw, idx, col_idx)

        if record["Original Text"].strip():
            rows.append(record)

    return pd.DataFrame(rows)


def cosine_sim(text_a: str, text_b: str) -> float:
    if not text_a.strip() and not text_b.strip():
        return 1.0
    if not text_a.strip() or not text_b.strip():
        return 0.0

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        preprocessor=None,
        lowercase=False,
    )
    matrix = vectorizer.fit_transform([text_a, text_b])
    return float(cosine_similarity(matrix[0], matrix[1])[0, 0])


def jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def lcs_similarity(text_a: str, text_b: str) -> float:
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0

    previous = [0] * (len(text_b) + 1)
    for char_a in text_a:
        current = [0]
        for j, char_b in enumerate(text_b, start=1):
            if char_a == char_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current

    return previous[-1] / max(len(text_a), len(text_b))


def sequence_similarity(text_a: str, text_b: str) -> float:
    return SequenceMatcher(None, text_a, text_b).ratio()


def levenshtein_distance(text_a: str, text_b: str) -> int:
    if Levenshtein is not None:
        return int(Levenshtein.distance(text_a, text_b))

    previous = list(range(len(text_b) + 1))
    for i, char_a in enumerate(text_a, start=1):
        current = [i]
        for j, char_b in enumerate(text_b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (char_a != char_b)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def edit_distance_similarity(text_a: str, text_b: str) -> float:
    max_len = max(len(text_a), len(text_b))
    if max_len == 0:
        return 1.0
    return 1 - (levenshtein_distance(text_a, text_b) / max_len)


def word_overlap_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))


def compute_metrics(text_a: str, text_b: str) -> dict[str, float]:
    return {
        "Cosine": cosine_sim(text_a, text_b),
        "Jaccard": jaccard_similarity(text_a, text_b),
        "LCS": lcs_similarity(text_a, text_b),
        "Sequence": sequence_similarity(text_a, text_b),
        "EditDist": edit_distance_similarity(text_a, text_b),
        "WordOverlap": word_overlap_similarity(text_a, text_b),
    }


def build_cleaned_dataset(raw_pairs: pd.DataFrame, preprocessor: BanglaPreprocessor) -> pd.DataFrame:
    cleaned = raw_pairs.copy()
    text_columns = ["Original Text"] + [f"{variation} Text" for variation, _ in VARIATION_COLUMNS]

    for column in text_columns:
        clean_column = column.replace(" Text", " Clean")
        token_count_column = column.replace(" Text", " Token Count")
        cleaned[clean_column] = cleaned[column].map(preprocessor.preprocess_text)
        cleaned[token_count_column] = cleaned[clean_column].map(lambda value: len(value.split()))

    return cleaned


def build_metrics(cleaned_pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, row in cleaned_pairs.iterrows():
        for variation, _ in VARIATION_COLUMNS:
            before = compute_metrics(row["Original Text"], row[f"{variation} Text"])
            after = compute_metrics(row["Original Clean"], row[f"{variation} Clean"])

            for stage, scores in [("Before", before), ("After", after)]:
                record: dict[str, object] = {
                    "Stage": stage,
                    "Test Case No": row["Test Case No"],
                    "Variation": variation,
                }
                record.update({metric: round(value, 6) for metric, value in scores.items()})
                rows.append(record)

    return pd.DataFrame(rows)


def build_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics_df.groupby(["Stage", "Variation"], as_index=False)[METRICS].mean()
    before = grouped[grouped["Stage"] == "Before"].set_index("Variation")[METRICS]
    after = grouped[grouped["Stage"] == "After"].set_index("Variation")[METRICS]
    delta = after - before

    summary = before.add_prefix("Before ").join(after.add_prefix("After "))
    summary = summary.join(delta.add_prefix("Delta "))
    summary = summary.reset_index()

    ordered_columns = ["Variation"]
    for metric in METRICS:
        ordered_columns.extend([f"Before {metric}", f"After {metric}", f"Delta {metric}"])
    return summary[ordered_columns].round(6)


def build_positive_recall(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Recall-at-threshold for positive controlled-plagiarism pairs.

    The current workbook contains controlled plagiarism variants only. This
    table is therefore recall-like sensitivity, not precision/F1.
    """

    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
    rows: list[dict[str, object]] = []

    for stage in ["Before", "After"]:
        stage_df = metrics_df[metrics_df["Stage"] == stage]
        for variation in [name for name, _ in VARIATION_COLUMNS]:
            variation_df = stage_df[stage_df["Variation"] == variation]
            for metric in METRICS:
                for threshold in thresholds:
                    rows.append(
                        {
                            "Stage": stage,
                            "Variation": variation,
                            "Metric": metric,
                            "Threshold": threshold,
                            "Positive Recall": round(float((variation_df[metric] >= threshold).mean()), 6),
                        }
                    )

    return pd.DataFrame(rows)


def build_wide_metrics(metrics_df: pd.DataFrame, stage: str) -> pd.DataFrame:
    stage_df = metrics_df[metrics_df["Stage"] == stage]
    rows: list[dict[str, object]] = []

    for case_id, case_df in stage_df.groupby("Test Case No", sort=False):
        record: dict[str, object] = {"Test Case No": case_id}
        for variation, _ in VARIATION_COLUMNS:
            variation_scores = case_df[case_df["Variation"] == variation]
            if variation_scores.empty:
                continue
            score_row = variation_scores.iloc[0]
            for metric in METRICS:
                record[f"{metric} ({variation})"] = score_row[metric]
        rows.append(record)

    return pd.DataFrame(rows)


def write_outputs(
    cleaned_pairs: pd.DataFrame,
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    recall_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = output_dir / "polished_dataset.xlsx"
    polished_textcase_path = output_dir / "polished_textcase.xlsx"
    metrics_path = output_dir / "before_after_similarity.xlsx"
    summary_path = output_dir / "summary_metrics.csv"
    long_csv_path = output_dir / "pair_metrics_long.csv"

    cleaned_pairs.to_excel(cleaned_path, index=False, engine="openpyxl")
    write_polished_textcase(cleaned_pairs, polished_textcase_path)
    metrics_df.to_csv(long_csv_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(metrics_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="pair_metrics_long", index=False)
        build_wide_metrics(metrics_df, "Before").to_excel(writer, sheet_name="wide_before", index=False)
        build_wide_metrics(metrics_df, "After").to_excel(writer, sheet_name="wide_after", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        recall_df.to_excel(writer, sheet_name="positive_recall", index=False)


def write_polished_textcase(cleaned_pairs: pd.DataFrame, output_path: Path) -> None:
    """Write cleaned text in the original row/column layout expected by similarity.py."""

    columns = [
        "Test Case No",
        "Original Clean",
        "Morphological Clean",
        "Order Clean",
        "Synonymy Clean",
        "Lexical Clean",
        "Mixed Clean",
    ]
    out_rows = [["", "", "", "", "", "", ""] for _ in range(4)]
    out_rows.append(
        [
            "",
            "Original text",
            "Morphological",
            "Order of word",
            "Synonymy",
            "Lexical variation",
            "Mixed",
        ]
    )

    for _, row in cleaned_pairs.iterrows():
        out_rows.append([row[column] for column in columns])

    pd.DataFrame(out_rows).to_excel(output_path, header=False, index=False, engine="openpyxl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Bangla plagiarism-pair data and compute before-vs-after similarity metrics."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input .xlsx workbook")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for outputs")
    parser.add_argument("--start-row", type=int, default=6, help="1-indexed first data row in Excel")
    parser.add_argument(
        "--end-row",
        type=int,
        default=105,
        help="1-indexed final data row in Excel. Use 0 to read to workbook end.",
    )
    parser.add_argument("--lexical-map", type=Path, default=DEFAULT_LEXICAL_MAP, help="Lexical TSV resource")
    parser.add_argument("--stopwords", type=Path, default=DEFAULT_STOPWORDS, help="Stopword text file")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_DEFAULTS),
        default="lexical_safe",
        help=(
            "Preprocessing profile. Use 'full' to enable stopword removal and "
            "stemming; use 'basic' for only Unicode/punctuation/lowercase/digit cleanup."
        ),
    )
    parser.add_argument(
        "--min-lexical-confidence",
        type=float,
        default=0.85,
        help="Only apply lexical-map rows with confidence >= this value",
    )
    parser.add_argument("--no-lexical-map", action="store_true", help="Disable lexical normalization")
    parser.add_argument("--use-stopwords", action="store_true", help="Enable stopword removal")
    parser.add_argument("--no-stopwords", action="store_true", help="Disable stopword removal")
    parser.add_argument("--use-stemming", action="store_true", help="Enable rule-based stemming")
    parser.add_argument("--no-stemming", action="store_true", help="Disable rule-based stemming")
    parser.add_argument("--keep-punctuation", action="store_true", help="Keep punctuation during preprocessing")
    parser.add_argument("--keep-native-digits", action="store_true", help="Do not normalize Bangla digits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_row = None if args.end_row == 0 else args.end_row
    profile = dict(PROFILE_DEFAULTS[args.profile])

    if args.no_lexical_map:
        profile["apply_lexical_mapping"] = False
    if args.use_stopwords:
        profile["remove_stopwords"] = True
    if args.no_stopwords:
        profile["remove_stopwords"] = False
    if args.use_stemming:
        profile["apply_stemming"] = True
    if args.no_stemming:
        profile["apply_stemming"] = False

    config = PreprocessConfig(
        normalize_digits=not args.keep_native_digits,
        remove_punctuation=not args.keep_punctuation,
        apply_lexical_mapping=profile["apply_lexical_mapping"],
        remove_stopwords=profile["remove_stopwords"],
        apply_stemming=profile["apply_stemming"],
        min_lexical_confidence=args.min_lexical_confidence,
    )

    preprocessor = BanglaPreprocessor(
        config,
        lexical_map_path=args.lexical_map,
        stopword_path=args.stopwords,
    )

    pairs = read_controlled_pairs(args.input, start_row=args.start_row, end_row=end_row)
    cleaned_pairs = build_cleaned_dataset(pairs, preprocessor)
    metrics_df = build_metrics(cleaned_pairs)
    summary_df = build_summary(metrics_df)
    recall_df = build_positive_recall(metrics_df)
    write_outputs(cleaned_pairs, metrics_df, summary_df, recall_df, args.output_dir)

    print(f"Rows processed: {len(cleaned_pairs)}")
    print(f"Outputs written to: {args.output_dir}")
    print(
        "Profile: "
        f"{args.profile} "
        f"(lexical={config.apply_lexical_mapping}, "
        f"stopwords={config.remove_stopwords}, "
        f"stemming={config.apply_stemming})"
    )
    print()
    print(summary_df.to_string(index=False))
    print()
    print(
        "Note: the current workbook contains positive controlled-plagiarism pairs only; "
        "precision, accuracy, and F1 require negative/non-plagiarized pairs."
    )


if __name__ == "__main__":
    main()
