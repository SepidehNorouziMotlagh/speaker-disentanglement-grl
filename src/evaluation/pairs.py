"""Matched speaker-verification pair generation.

Controls for emotion and sentence content so the attacker cannot solve
the verification task using non-speaker cues.

Positive pair: same speaker, different sentence IDs, same emotion.
Negative pair: different speakers, same two sentence IDs, same emotion.

Known limitation: intensity is not matched.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def generate_matched_pairs(
    embeddings: np.ndarray,
    speakers: np.ndarray,
    sentences: np.ndarray,
    emotions: np.ndarray,
    n_pairs: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    # groups[emotion][sentence][speaker] -> list of utterance indices
    groups: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for idx, (speaker, sentence, emotion) in enumerate(
        zip(speakers, sentences, emotions)
    ):
        groups[emotion][sentence][speaker].append(idx)

    positive_by_key: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    negative_by_key: dict[tuple, list[tuple[int, int]]] = defaultdict(list)

    for emotion, sentence_groups in groups.items():
        sentence_ids = sorted(sentence_groups)
        for i, sentence_a in enumerate(sentence_ids):
            for sentence_b in sentence_ids[i + 1 :]:
                speakers_a = sentence_groups[sentence_a]
                speakers_b = sentence_groups[sentence_b]

                for speaker in set(speakers_a) & set(speakers_b):
                    for idx_a in speakers_a[speaker]:
                        for idx_b in speakers_b[speaker]:
                            positive_by_key[(emotion, sentence_a, sentence_b)].append(
                                (idx_a, idx_b)
                            )

                for speaker_a, indices_a in speakers_a.items():
                    for speaker_b, indices_b in speakers_b.items():
                        if speaker_a == speaker_b:
                            continue
                        for idx_a in indices_a:
                            for idx_b in indices_b:
                                negative_by_key[(emotion, sentence_a, sentence_b)].append(
                                    (idx_a, idx_b)
                                )

    common_keys = sorted(set(positive_by_key) & set(negative_by_key))
    if not common_keys:
        raise ValueError("No matched positive/negative pair groups found.")

    rng.shuffle(common_keys)
    positive_pairs: list[tuple[int, int]] = []
    negative_pairs: list[tuple[int, int]] = []
    per_key = max(1, n_pairs // len(common_keys))

    for key in common_keys:
        pos = positive_by_key[key]
        neg = negative_by_key[key]
        n = min(per_key, len(pos), len(neg))
        if n == 0:
            continue
        pos_idx = rng.choice(len(pos), size=n, replace=False)
        neg_idx = rng.choice(len(neg), size=n, replace=False)
        positive_pairs.extend([pos[i] for i in pos_idx])
        negative_pairs.extend([neg[i] for i in neg_idx])

    n = min(len(positive_pairs), len(negative_pairs), n_pairs)
    if n == 0:
        raise ValueError("No balanced matched pairs could be generated.")

    pos_idx = rng.choice(len(positive_pairs), size=n, replace=False)
    neg_idx = rng.choice(len(negative_pairs), size=n, replace=False)
    pairs = np.array(
        [positive_pairs[i] for i in pos_idx] + [negative_pairs[i] for i in neg_idx]
    )
    labels = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)])
    order = rng.permutation(len(labels))
    pairs = pairs[order]
    labels = labels[order]

    features = np.abs(embeddings[pairs[:, 0]] - embeddings[pairs[:, 1]]).astype(
        np.float32
    )
    return features, labels