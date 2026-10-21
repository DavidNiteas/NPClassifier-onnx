"""Replication of the official NPClassifier decision logic.

The official web application does not simply threshold every class at 0.5.
It thresholds the three sigmoid heads at different levels, then votes across
the ontology hierarchy:

* class: 0.1
* superclass: 0.3
* pathway: 0.5

The thresholds and voting rules below are a direct port of the official
``Classifier/prediction_voting.py`` and ``app.py`` logic.  The model output is
still a set of per-class sigmoid scores; this module only turns those scores
into the official result lists.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np
from rdkit import Chem

OFFICIAL_THRESHOLDS: Dict[str, float] = {
    "class": 0.1,
    "superclass": 0.3,
    "pathway": 0.5,
}

# The production app uses Classifier/fingerprint_handler._isglycoside.
_GLYCOSIDE_PATTERNS = (
    Chem.MolFromSmarts(
        "[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),"
        "$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]"
    ),
    Chem.MolFromSmarts(
        "[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C1),"
        "$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C(O)@C1)]"
    ),
    Chem.MolFromSmarts(
        "[OX2;$([r5]1@C(!@[OX2H1])@C@C@C1),"
        "$([r6]1@C(!@[OX2H1])@C@C@C@C1)]"
    ),
)


def is_glycoside(smiles: str) -> Union[bool, str]:
    """Official glycoside SMARTS check.

    Returns ``True``/``False`` for a parsed molecule and ``"Input_error"`` for
    an invalid SMILES, matching the official helper.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Input_error"
    return any(mol.HasSubstructMatch(pattern) for pattern in _GLYCOSIDE_PATTERNS)


def _unique(values: Iterable[int]) -> List[int]:
    """Return unique integers in ascending order (stable, reproducible)."""
    return sorted({int(value) for value in values})


def _labels(indices: Iterable[int], label_list: Sequence[str]) -> List[str]:
    return [str(label_list[int(index)]) for index in _unique(indices)]


def _paths_from_classes(
    class_ids: Sequence[int],
    ontology: Mapping[str, dict],
) -> List[int]:
    paths: List[int] = []
    hierarchy = ontology["Class_hierarchy"]
    for class_id in class_ids:
        paths.extend(hierarchy[str(int(class_id))]["Pathway"])
    return _unique(paths)


def _paths_from_superclasses(
    superclass_ids: Sequence[int],
    ontology: Mapping[str, dict],
) -> List[int]:
    paths: List[int] = []
    hierarchy = ontology["Super_hierarchy"]
    for superclass_id in superclass_ids:
        paths.extend(hierarchy[str(int(superclass_id))]["Pathway"])
    return _unique(paths)


def _superclasses_from_classes(
    class_ids: Sequence[int],
    ontology: Mapping[str, dict],
) -> List[int]:
    hierarchy = ontology["Class_hierarchy"]
    values: List[int] = []
    for class_id in class_ids:
        values.extend(hierarchy[str(int(class_id))]["Superclass"])
    return _unique(values)


def _paths_from_class_ids(
    class_ids: Sequence[int],
    ontology: Mapping[str, dict],
) -> List[int]:
    hierarchy = ontology["Class_hierarchy"]
    values: List[int] = []
    for class_id in class_ids:
        values.extend(hierarchy[str(int(class_id))]["Pathway"])
    return _unique(values)


def vote_classification(
    n_path: Sequence[int],
    n_class: Sequence[int],
    n_super: Sequence[int],
    pred_class: np.ndarray,
    pred_super: np.ndarray,
    path_from_class: Sequence[int],
    path_from_superclass: Sequence[int],
    isglycoside: Union[bool, str],
    ontology: Mapping[str, dict],
) -> Tuple[List[str], List[str], List[str], Union[bool, str]]:
    """Port of ``Classifier/prediction_voting.py::vote_classification``.

    All indices refer to the order of the label dictionaries in the official
    ontology (``index_v1.json``): ``Class``, ``Superclass`` and ``Pathway``.
    The returned lists are deterministic: IDs are de-duplicated and sorted.
    """
    index_class = list(ontology["Class"].keys())
    index_superclass = list(ontology["Superclass"].keys())
    index_pathway = list(ontology["Pathway"].keys())
    class_hierarchy = ontology["Class_hierarchy"]
    super_hierarchy = ontology["Super_hierarchy"]

    n_path = _unique(n_path)
    n_class = _unique(n_class)
    n_super = _unique(n_super)
    path_from_class = _unique(path_from_class)
    path_from_superclass = _unique(path_from_superclass)

    path_for_vote = list(n_path) + list(path_from_class) + list(path_from_superclass)
    path = _unique(
        value for value in path_for_vote if path_for_vote.count(value) == 3
    )

    if not path:
        path = _unique(
            value for value in path_for_vote if path_for_vote.count(value) == 2
        )
        # The official code re-computes the same set here when len(path) > 1.
        # It has no effect, so it is intentionally omitted.
    if not path:
        return (
            _labels(n_path, index_pathway),
            [],
            [],
            isglycoside,
        )

    if set(n_path) & set(path):
        if set(path) & set(path_from_superclass):
            n_super = [
                value
                for value in n_super
                if set(path) & set(super_hierarchy[str(value)]["Pathway"])
            ]
            if not n_super:
                n_class = [
                    value
                    for value in n_class
                    if set(path) & set(class_hierarchy[str(value)]["Pathway"])
                ]
                n_super = _superclasses_from_classes(n_class, ontology)
            elif len(n_super) > 1:
                n_class = [
                    value
                    for value in n_class
                    if set(path) & set(class_hierarchy[str(value)]["Pathway"])
                ]
                if n_class:
                    n_super = _superclasses_from_classes(n_class, ontology)
                    n_path = _paths_from_class_ids(n_class, ontology)
                elif len(path) == 1:
                    n_super = [int(np.argmax(pred_super))]
                    n_class = [
                        int(np.argmax(pred_class))
                    ] if set(n_super) & set(
                        class_hierarchy[str(int(np.argmax(pred_class)))]["Superclass"]
                    ) else []
            else:
                n_class = [
                    value
                    for value in n_class
                    if set(n_super)
                    & set(class_hierarchy[str(value)]["Superclass"])
                ]
                if not n_class:
                    candidate = int(np.argmax(pred_class))
                    if set(n_super) & set(
                        class_hierarchy[str(candidate)]["Superclass"]
                    ):
                        n_class = [candidate]
        else:
            n_class = [
                value
                for value in n_class
                if set(path) & set(class_hierarchy[str(value)]["Pathway"])
            ]
            n_super = _superclasses_from_classes(n_class, ontology)
    else:
        n_super = [
            value
            for value in n_super
            if set(path) & set(super_hierarchy[str(value)]["Pathway"])
        ]
        if not n_super:
            n_class = [
                value
                for value in n_class
                if set(path) & set(class_hierarchy[str(value)]["Pathway"])
            ]
            n_super = _superclasses_from_classes(n_class, ontology)
            n_path = _paths_from_class_ids(n_class, ontology)
        elif len(n_super) > 1:
            n_class = [
                value
                for value in n_class
                if set(path) & set(class_hierarchy[str(value)]["Pathway"])
            ]
            n_super = _superclasses_from_classes(n_class, ontology)
            n_path = _paths_from_class_ids(n_class, ontology)
        else:
            n_class = [
                value
                for value in n_class
                if set(path) & set(class_hierarchy[str(value)]["Pathway"])
            ]
            n_super = _superclasses_from_classes(n_class, ontology)
            n_path = _paths_from_class_ids(n_class, ontology)

    return (
        _labels(path, index_pathway),
        _labels(n_super, index_superclass),
        _labels(n_class, index_class),
        isglycoside,
    )
