from __future__ import annotations

import numpy as np
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


CLASSIFIER_ALIASES = {
    "rf": "rf",
    "random_forest": "rf",
    "randomforest": "rf",
    "extratrees": "extratrees",
    "extra_trees": "extratrees",
    "bagged_cart": "bagged_cart",
    "baggedcart": "bagged_cart",
    "random_subspace_trees": "random_subspace_trees",
    "rst": "random_subspace_trees",
    "decision_tree": "decision_tree",
    "cart": "decision_tree",
}


def canonical_classifier_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in CLASSIFIER_ALIASES:
        raise ValueError(f"unknown classifier {name!r}; choose from {sorted(set(CLASSIFIER_ALIASES.values()))}")
    return CLASSIFIER_ALIASES[key]


def make_classifier(name: str, *, random_state: int = 42, n_estimators: int = 200, n_jobs: int = 1, weighted: bool = False):
    name = canonical_classifier_name(name)
    if name == "rf":
        kwargs = {"class_weight": "balanced"} if weighted else {}
        return RandomForestClassifier(n_estimators=int(n_estimators), random_state=int(random_state), n_jobs=int(n_jobs), **kwargs)
    if name == "extratrees":
        kwargs = {"class_weight": "balanced"} if weighted else {}
        return ExtraTreesClassifier(n_estimators=int(n_estimators), random_state=int(random_state), n_jobs=int(n_jobs), **kwargs)
    if name == "bagged_cart":
        tree_kwargs = {"class_weight": "balanced"} if weighted else {}
        base = DecisionTreeClassifier(random_state=int(random_state), **tree_kwargs)
        return BaggingClassifier(
            estimator=base,
            n_estimators=int(n_estimators),
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            random_state=int(random_state),
            n_jobs=int(n_jobs),
        )
    if name == "random_subspace_trees":
        tree_kwargs = {"class_weight": "balanced"} if weighted else {}
        base = DecisionTreeClassifier(random_state=int(random_state), **tree_kwargs)
        return BaggingClassifier(
            estimator=base,
            n_estimators=int(n_estimators),
            max_samples=1.0,
            max_features=0.5,
            bootstrap=False,
            bootstrap_features=False,
            random_state=int(random_state),
            n_jobs=int(n_jobs),
        )
    if name == "decision_tree":
        kwargs = {"class_weight": "balanced"} if weighted else {}
        return DecisionTreeClassifier(random_state=int(random_state), **kwargs)
    raise ValueError(f"unknown classifier {name!r}")


def make_rival_classifier(name: str, *, random_state: int = 42, n_estimators: int = 200, n_jobs: int = 1):
    name = str(name).strip().lower()
    if name == "balanced_random_forest":
        from imblearn.ensemble import BalancedRandomForestClassifier

        return BalancedRandomForestClassifier(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
            n_jobs=int(n_jobs),
            sampling_strategy="all",
            replacement=True,
            bootstrap=False,
        )
    if name == "balanced_bagging":
        from imblearn.ensemble import BalancedBaggingClassifier

        return BalancedBaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=int(random_state)),
            n_estimators=int(n_estimators),
            random_state=int(random_state),
            n_jobs=int(n_jobs),
            sampling_strategy="auto",
            replacement=True,
            bootstrap=False,
        )
    if name == "easy_ensemble":
        from imblearn.ensemble import EasyEnsembleClassifier

        return EasyEnsembleClassifier(n_estimators=20, random_state=int(random_state), n_jobs=int(n_jobs))
    if name == "rus_boost":
        from imblearn.ensemble import RUSBoostClassifier

        return RUSBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=int(random_state)),
            n_estimators=int(n_estimators),
            learning_rate=0.5,
            random_state=int(random_state),
            replacement=True,
        )
    raise ValueError(f"unknown rival classifier {name!r}")


RIVAL_METHODS = ["balanced_random_forest", "balanced_bagging", "easy_ensemble", "rus_boost"]
