import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_utils import find_best_threshold


class TestFindBestThreshold:
    # Vérifie que la fonction retourne un tuple (seuil, coût)
    def test_returns_tuple(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        result = find_best_threshold(y_true, y_prob)
        assert isinstance(result, tuple)
        assert len(result) == 2

    # Vérifie que le seuil optimal est dans la plage valide [0.01, 1.0)
    def test_threshold_in_valid_range(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        threshold, cost = find_best_threshold(y_true, y_prob)
        assert 0.01 <= threshold < 1.0

    # Vérifie que le coût métier est toujours positif ou nul
    def test_cost_is_non_negative(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        threshold, cost = find_best_threshold(y_true, y_prob)
        assert cost >= 0

    # Vérifie que le coût est nul quand les classes sont parfaitement séparées
    def test_perfect_separation_low_cost(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        threshold, cost = find_best_threshold(y_true, y_prob)
        assert cost == 0
