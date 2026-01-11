import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from metrics import (
    calculate_auc,
    calculate_recall,
    calculate_precision,
    calculate_f1,
    calculate_accuracy,
    business_cost_metric
)
from config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE


class TestCalculateAuc:
    # Vérifie que l'AUC est parfaite (1.0) quand les prédictions séparent bien les classes
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9])
        assert calculate_auc(y_true, y_pred_proba) == 1.0

    # Vérifie que l'AUC est de 0.5 quand les prédictions sont aléatoires
    def test_random_predictions(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([0.5, 0.5, 0.5, 0.5])
        assert calculate_auc(y_true, y_pred_proba) == 0.5

    # Vérifie que la fonction retourne NaN pour un tableau vide
    def test_empty_array_returns_nan(self):
        y_true = np.array([])
        y_pred_proba = np.array([])
        result = calculate_auc(y_true, y_pred_proba)
        assert np.isnan(result)


class TestCalculateRecall:
    # Vérifie que le rappel est parfait quand tous les positifs sont détectés
    def test_perfect_recall(self):
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([1, 1, 1, 0])
        assert calculate_recall(y_true, y_pred) == 1.0

    # Vérifie que le rappel est nul quand aucun positif n'est détecté
    def test_zero_recall(self):
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([0, 0, 0, 0])
        assert calculate_recall(y_true, y_pred) == 0.0


class TestBusinessCostMetric:
    # Vérifie que le coût est nul quand il n'y a aucune erreur
    def test_no_errors(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9])
        cost = business_cost_metric(y_true, y_pred_proba, threshold=0.5)
        assert cost == 0

    # Vérifie le coût quand tous les défauts sont manqués (faux négatifs)
    def test_all_false_negatives(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.1, 0.1, 0.1])
        cost = business_cost_metric(y_true, y_pred_proba, threshold=0.5)
        assert cost == 4 * COST_FALSE_NEGATIVE

    # Vérifie le coût quand tous les bons clients sont refusés (faux positifs)
    def test_all_false_positives(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred_proba = np.array([0.9, 0.9, 0.9, 0.9])
        cost = business_cost_metric(y_true, y_pred_proba, threshold=0.5)
        assert cost == 4 * COST_FALSE_POSITIVE

    # Vérifie le coût avec un mélange de faux positifs et faux négatifs
    def test_mixed_errors(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred_proba = np.array([0.3, 0.7, 0.8, 0.2])
        cost = business_cost_metric(y_true, y_pred_proba, threshold=0.5)
        expected_cost = 1 * COST_FALSE_NEGATIVE + 1 * COST_FALSE_POSITIVE
        assert cost == expected_cost

    # Vérifie que la fonction retourne NaN pour un tableau vide
    def test_empty_array_returns_nan(self):
        y_true = np.array([])
        y_pred_proba = np.array([])
        result = business_cost_metric(y_true, y_pred_proba)
        assert np.isnan(result)

    # Vérifie que le coût change en fonction du seuil choisi
    def test_threshold_sensitivity(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred_proba = np.array([0.6, 0.4, 0.6, 0.4])
        
        cost_low = business_cost_metric(y_true, y_pred_proba, threshold=0.3)
        cost_high = business_cost_metric(y_true, y_pred_proba, threshold=0.7)
        
        assert cost_low != cost_high
