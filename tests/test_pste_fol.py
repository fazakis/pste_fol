from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from pste_fol import FastOutwardLadderOversampler, PSTEClassifier
from pste_fol.datasets import (
    AUXILIARY_CATEGORY7_DATASETS,
    PRIMARY_NUMERIC18_DATASETS,
    DatasetRecord,
    resolve_dataset_names,
)
from pste_fol.experiment import (
    PAPER_RETAINED_OVERSAMPLERS,
    run_experiment,
)
from pste_fol.reference import (
    METHODS,
    focused_pairwise,
    load_reference_results,
    rank_summary,
    task_matrix,
)


class PSTEFOLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_classification(
            n_samples=180,
            n_features=8,
            n_informative=5,
            n_redundant=1,
            weights=[0.82, 0.18],
            class_sep=0.9,
            random_state=7,
        )
        cls.X = np.asarray(X, dtype=float)
        cls.y = np.asarray(y, dtype=int)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X,
            cls.y,
            test_size=0.25,
            random_state=11,
            stratify=cls.y,
        )

    def test_fol_is_deterministic_and_ladder_points_pass_strict_safety(self):
        first = FastOutwardLadderOversampler(random_state=42)
        second = FastOutwardLadderOversampler(random_state=42)
        X_first, y_first = first.fit_resample(self.X_train, self.y_train)
        X_second, y_second = second.fit_resample(self.X_train, self.y_train)

        np.testing.assert_array_equal(X_first, X_second)
        np.testing.assert_array_equal(y_first, y_second)
        self.assertEqual(first.n_generated_, second.n_generated_)
        self.assertEqual(first.n_selected_ladder_, len(first.selected_ladder_points_))

        if len(first.selected_ladder_points_):
            minority = self.X_train[self.y_train == 1]
            majority = self.X_train[self.y_train == 0]
            min_nn = NearestNeighbors(n_neighbors=1).fit(minority)
            maj_nn = NearestNeighbors(n_neighbors=1).fit(majority)
            d_min = min_nn.kneighbors(
                first.selected_ladder_points_,
                return_distance=True,
            )[0][:, 0]
            d_maj = maj_nn.kneighbors(
                first.selected_ladder_points_,
                return_distance=True,
            )[0][:, 0]
            self.assertTrue(np.all(d_maj > d_min))

        digest = hashlib.sha256()
        for array in (
            np.ascontiguousarray(X_first),
            np.ascontiguousarray(y_first),
        ):
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        self.assertEqual(
            digest.hexdigest(),
            "0e1850a16c9e03bc9a9b576e4dbe538428068aef272f58a2d7b1703f298e35ed",
        )
        self.assertEqual(first.n_generated_, 85)
        self.assertEqual(first.n_base_smote_, 34)
        self.assertEqual(first.n_ladder_candidates_, 97)
        self.assertEqual(first.n_selected_ladder_, 51)
        self.assertEqual(first.n_rejected_ladder_, 11)

    def test_pste_uses_fixed_budget_and_no_inner_validation(self):
        base = RandomForestClassifier(
            n_estimators=30,
            random_state=42,
            n_jobs=1,
        )
        estimator = PSTEClassifier(
            base,
            total_estimators=30,
            random_state=42,
            n_jobs=1,
        ).fit(self.X_train, self.y_train)

        self.assertEqual(estimator.original_budget_, 20)
        self.assertEqual(estimator.shadow_budget_, 10)
        self.assertEqual(estimator.inner_cv_splits_, 0)
        self.assertEqual(
            estimator.selection_reason_,
            "fixed_protocol_no_inner_validation",
        )
        self.assertEqual(
            estimator.n_resampled_,
            estimator.n_shadow_train_,
        )
        self.assertEqual(
            estimator.total_branch_training_rows_,
            len(self.y_train) + estimator.n_shadow_train_,
        )
        self.assertTrue(np.isfinite(estimator.support_scale_))

        components = estimator.predict_components(self.X_test)
        probability = estimator.predict_proba(self.X_test)
        self.assertEqual(probability.shape, (len(self.X_test), 2))
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)
        self.assertTrue(np.all((components["gate"] >= 0.0) & (components["gate"] <= 1.0)))
        bounded_move = (
            estimator.correction_strength
            * components["gate"]
            * components["local_shift"]
        )
        self.assertLessEqual(float(np.max(np.abs(bounded_move))), 0.3 + 1e-12)

        # The gate scale is learned during fit. A query's prediction must not
        # depend on which unrelated rows share the prediction batch.
        one = estimator.predict_proba(self.X_test[:1])[:, 1]
        batch = estimator.predict_proba(self.X_test)[:1, 1]
        np.testing.assert_array_equal(one, batch)

        digest = hashlib.sha256()
        for key in (
            "original",
            "shadow_raw",
            "shadow_corrected",
            "local_shift",
            "support_gate",
            "class_confidence",
            "gate",
            "final",
        ):
            array = np.ascontiguousarray(components[key])
            digest.update(key.encode())
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        self.assertEqual(
            digest.hexdigest(),
            "7c13717c002b2f4c964dd61860e0f54aff91541f966fa4790fe8aa2c072a0e22",
        )

    def test_small_experiment_runs_end_to_end(self):
        record = DatasetRecord(
            X=self.X,
            y=self.y,
            name="synthetic-smoke",
            source="unit-test",
            metadata={},
        )
        results = run_experiment(
            [record],
            seeds=[42],
            folds=2,
            classifiers=["rf"],
            oversamplers=["fast_outward_ladder"],
            method_groups=["native", "oversampler", "pste"],
            total_estimators=20,
            classifier_n_jobs=1,
        )
        self.assertEqual(len(results), 6)
        self.assertEqual(
            set(results["method"]),
            {"rf_none", "rf_fast_outward_ladder", "pste_rf_fol"},
        )
        pste_rows = results[results["method"].eq("pste_rf_fol")]
        self.assertTrue(
            pste_rows["estimators_by_component"].eq("13+7").all()
        )
        self.assertTrue(
            pste_rows["notes"].str.contains("no inner validation").all()
        )

    def test_locked_primary_scope_is_18_tasks_and_22_methods(self):
        self.assertEqual(len(PRIMARY_NUMERIC18_DATASETS), 18)
        self.assertEqual(len(AUXILIARY_CATEGORY7_DATASETS), 7)
        self.assertFalse(
            set(PRIMARY_NUMERIC18_DATASETS)
            & set(AUXILIARY_CATEGORY7_DATASETS)
        )
        self.assertEqual(resolve_dataset_names(["paper"]), PRIMARY_NUMERIC18_DATASETS)
        self.assertEqual(len(PAPER_RETAINED_OVERSAMPLERS), 8)
        self.assertIn("geometric_smote", PAPER_RETAINED_OVERSAMPLERS)
        self.assertNotIn("deep_smote", PAPER_RETAINED_OVERSAMPLERS)
        self.assertEqual(len(METHODS), 22)

    def test_locked_primary_scheduler_emits_22_methods_per_backbone(self):
        record = DatasetRecord(
            X=self.X,
            y=self.y,
            name="synthetic-scheduler",
            source="unit-test",
            metadata={},
        )

        def result_for(X_test, y_train):
            score = np.full(len(X_test), 0.25, dtype=float)
            return (
                np.zeros(len(X_test), dtype=int),
                score,
                len(y_train),
                0,
                "",
                "test-component",
                "200",
                "scheduler test",
            )

        def fake_native(_classifier, _X_train, y_train, X_test, *_args, **_kwargs):
            return result_for(X_test, y_train)

        sampled_kwargs = []

        def fake_sampled(
            _classifier,
            _oversampler,
            _X_train,
            y_train,
            X_test,
            *_args,
            **_kwargs,
        ):
            sampled_kwargs.append(_kwargs)
            return result_for(X_test, y_train)

        def fake_rival(_method, _X_train, y_train, X_test, *_args, **_kwargs):
            return result_for(X_test, y_train)

        def fake_mgvae(X_train, y_train, **_kwargs):
            return (
                np.asarray(X_train, dtype=float),
                np.asarray(y_train, dtype=int),
                {
                    "n_generated": 0,
                    "warning": "",
                    "training_loss": 0.0,
                    "runtime_seconds": 0.0,
                    "epochs": 200,
                    "sampler_seed": 42,
                },
            )

        with (
            patch("pste_fol.experiment.fit_predict_native", side_effect=fake_native),
            patch(
                "pste_fol.experiment.fit_predict_oversampler",
                side_effect=fake_sampled,
            ),
            patch("pste_fol.experiment.fit_predict_pste", side_effect=fake_sampled),
            patch("pste_fol.experiment.fit_predict_rival", side_effect=fake_rival),
            patch(
                "pste_fol.experiment.build_shared_mgvae_shadow",
                side_effect=fake_mgvae,
            ) as mgvae,
        ):
            results = run_experiment(
                [record],
                seeds=[42],
                folds=2,
                paper_mode="kbs",
                classifier_n_jobs=1,
                oversampler_kwargs={"max_candidates": 1},
                mgvae_torch_threads=4,
            )

        self.assertEqual(len(results), 2 * 3 * 22)
        self.assertEqual(mgvae.call_count, 2)
        self.assertTrue(
            all(item["oversampler_kwargs"] == {} for item in sampled_kwargs)
        )
        self.assertTrue(
            all(
                call.kwargs["torch_threads"] == 1
                and call.kwargs["epochs"] == 200
                for call in mgvae.call_args_list
            )
        )
        self.assertEqual(set(results.logical_method), set(METHODS))
        counts = results.groupby(
            ["seed", "fold", "screen_backbone"]
        ).logical_method.nunique()
        self.assertTrue(counts.eq(22).all())

    def test_packaged_primary_reference_reconstructs_headlines(self):
        root = Path(__file__).resolve().parents[1]
        frame = load_reference_results(
            root / "reference" / "primary" / "validated_fold_metrics_17820.csv"
        )
        matrix = task_matrix(frame)
        ranking, summary = rank_summary(matrix)
        self.assertEqual(matrix.shape, (18, 22))
        self.assertEqual(ranking.iloc[0].logical_method, "pste_fol")
        self.assertAlmostEqual(
            float(ranking.iloc[0].mean_pr_auc),
            0.723399719029844,
            places=14,
        )
        self.assertAlmostEqual(
            float(ranking.iloc[0].average_rank),
            4.277777777777778,
            places=14,
        )
        self.assertTrue(
            ranking.head(8).logical_method.str.startswith("pste_").all()
        )
        self.assertAlmostEqual(
            float(summary["friedman_chi_square"]),
            204.59288537549423,
            places=12,
        )
        pairwise = focused_pairwise(matrix)
        fol = pairwise.loc[pairwise.method_b.eq("os_fol")].iloc[0]
        self.assertEqual(int(fol.wins_a), 15)
        self.assertAlmostEqual(float(fol.holm_p_21), 0.00295257568359375)


if __name__ == "__main__":
    unittest.main()
