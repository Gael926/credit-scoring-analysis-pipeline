import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_prep import create_domain_features


class TestCreateDomainFeatures:
    # Fixture pour créer un DataFrame de test avec des données réalistes
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'AMT_CREDIT': [100000, 200000, 150000],
            'AMT_INCOME_TOTAL': [50000, 100000, 75000],
            'AMT_ANNUITY': [5000, 10000, 7500],
            'DAYS_EMPLOYED': [-365, -730, -1095],
            'DAYS_BIRTH': [-10000, -15000, -12000],
            'AMT_GOODS_PRICE': [90000, 180000, 140000],
            'EXT_SOURCE_1': [0.5, 0.6, 0.7],
            'EXT_SOURCE_2': [0.4, 0.5, 0.6],
            'EXT_SOURCE_3': [0.3, 0.4, 0.5]
        })

    # Vérifie le calcul du ratio crédit/revenu
    def test_credit_income_percent(self, sample_df):
        result = create_domain_features(sample_df)
        expected = sample_df['AMT_CREDIT'] / sample_df['AMT_INCOME_TOTAL']
        pd.testing.assert_series_equal(result['CREDIT_INCOME_PERCENT'], expected, check_names=False)

    # Vérifie le calcul du taux d'endettement (annuité/revenu)
    def test_annuity_income_percent(self, sample_df):
        result = create_domain_features(sample_df)
        expected = sample_df['AMT_ANNUITY'] / sample_df['AMT_INCOME_TOTAL']
        pd.testing.assert_series_equal(result['ANNUITY_INCOME_PERCENT'], expected, check_names=False)

    # Vérifie le calcul de la durée du crédit
    def test_credit_term(self, sample_df):
        result = create_domain_features(sample_df)
        expected = sample_df['AMT_ANNUITY'] / sample_df['AMT_CREDIT']
        pd.testing.assert_series_equal(result['CREDIT_TERM'], expected, check_names=False)

    # Vérifie le calcul de la moyenne des scores externes
    def test_ext_source_mean(self, sample_df):
        result = create_domain_features(sample_df)
        expected = sample_df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        pd.testing.assert_series_equal(result['EXT_SOURCE_MEAN'], expected, check_names=False)

    # Vérifie que la division par zéro génère des valeurs infinies
    def test_division_by_zero_handling(self):
        df = pd.DataFrame({
            'AMT_CREDIT': [100000],
            'AMT_INCOME_TOTAL': [0],
            'AMT_ANNUITY': [5000],
            'DAYS_EMPLOYED': [-365],
            'DAYS_BIRTH': [-10000],
            'AMT_GOODS_PRICE': [0],
            'EXT_SOURCE_1': [0.5],
            'EXT_SOURCE_2': [0.4],
            'EXT_SOURCE_3': [0.3]
        })
        result = create_domain_features(df)
        assert np.isinf(result['CREDIT_INCOME_PERCENT'].iloc[0])
        assert np.isinf(result['CREDIT_TO_GOODS_RATIO'].iloc[0])

    # Vérifie que toutes les colonnes attendues sont créées
    def test_all_columns_created(self, sample_df):
        result = create_domain_features(sample_df)
        expected_columns = [
            'CREDIT_INCOME_PERCENT',
            'ANNUITY_INCOME_PERCENT',
            'CREDIT_TERM',
            'DAYS_EMPLOYED_PERCENT',
            'CREDIT_TO_GOODS_RATIO',
            'EXT_SOURCE_MEAN',
            'EXT_SOURCE_PROD',
            'EXT_SOURCE_1_x_DAYS_BIRTH'
        ]
        for col in expected_columns:
            assert col in result.columns
