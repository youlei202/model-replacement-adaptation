from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .frozen import require_pretest_fitting


TFIDF_MAX_FEATURES = 256
TFIDF_TOKEN_PATTERN = r"(?u)\b\w\w+\b"


def fit_tfidf_state(texts: Sequence[str]) -> Dict[str, object]:
    require_pretest_fitting()
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        dtype=np.float64,
        lowercase=True,
        token_pattern=TFIDF_TOKEN_PATTERN,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )
    vectorizer.fit([str(text) for text in texts])
    vocabulary = sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get)
    return {
        "state_family": "public_input_tfidf",
        "state_version": 1,
        "fit_split": "fit",
        "max_features": TFIDF_MAX_FEATURES,
        "lowercase": True,
        "token_pattern": TFIDF_TOKEN_PATTERN,
        "norm": "l2",
        "smooth_idf": True,
        "sublinear_tf": False,
        "vocabulary": vocabulary,
        "idf": vectorizer.idf_.astype(float).tolist(),
    }


def transform_tfidf_state(state: Dict[str, object], texts: Sequence[str]) -> np.ndarray:
    vocabulary = [str(token) for token in state["vocabulary"]]
    idf = np.asarray(state["idf"], dtype=np.float64)
    if not vocabulary or len(vocabulary) != len(idf):
        raise RuntimeError("frozen TF-IDF state has inconsistent dimensions")
    if state.get("state_family") != "public_input_tfidf":
        raise RuntimeError("unexpected frozen public-feature state")

    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    counter = CountVectorizer(
        vocabulary=token_to_index,
        dtype=np.float64,
        lowercase=bool(state["lowercase"]),
        token_pattern=str(state["token_pattern"]),
    )
    counts = counter.transform([str(text) for text in texts])
    weighted = counts.multiply(idf).tocsr()
    return normalize(weighted, norm=str(state["norm"]), copy=False).toarray()
