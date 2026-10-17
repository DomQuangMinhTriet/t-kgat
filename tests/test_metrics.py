import math

import numpy as np
import scipy.sparse as sp

from tkgat.eval.metrics import evaluate, recall_ndcg_at_k


def test_hand_computed_example():
    # ranking [5, 1, 9, 2], truth {1, 2, 7}
    r, n = recall_ndcg_at_k(np.array([5, 1, 9, 2]), np.array([1, 2, 7]), k=4)
    assert math.isclose(r, 2 / 3)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert math.isclose(n, dcg / idcg)


def test_seen_items_are_masked():
    # item 0 has the highest score but was already seen -> must not count
    scores = np.array([[10.0, 1.0, 5.0, 0.0]])
    mask = sp.csr_matrix(np.array([[1, 0, 0, 0]], dtype=np.float32))
    res = evaluate(lambda users: scores, {0: np.array([2])}, mask, ks=(1,))
    assert res["recall@1"] == 1.0


def test_perfect_and_empty_ranking():
    r, n = recall_ndcg_at_k(np.array([3, 4]), np.array([3, 4]), k=2)
    assert r == 1.0 and math.isclose(n, 1.0)
    r, n = recall_ndcg_at_k(np.array([0, 1]), np.array([3]), k=2)
    assert r == 0.0 and n == 0.0
