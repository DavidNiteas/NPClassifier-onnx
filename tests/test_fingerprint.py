from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from NPClassifier_onnx import NPClassifier

RDLogger.DisableLog("rdApp.*")


def legacy_fingerprint(smiles: str):
    """Reference implementation copied from the original NPClassifier wrapper."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    binary = np.zeros(4096, int)
    formula = np.zeros(2048, int)
    bit_info = {}
    for radius in range(3):
        fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=radius, bitInfo=bit_info, nBits=2048
        )
        selected = []
        for bit in fingerprint.GetOnBits():
            for environment in range(len(bit_info[bit])):
                if bit_info[bit][environment][1] == radius:
                    selected.append(bit)
                    break
        if radius == 0:
            for bit in selected:
                formula[bit] = len(
                    [item for item in bit_info[bit] if item[1] == 0]
                )
        else:
            for bit in selected:
                binary[2048 * (radius - 1) + bit] = len(
                    [item for item in bit_info[bit] if item[1] == radius]
                )
    return formula, binary


def test_fingerprint_matches_legacy_implementation():
    molecules = [
        "CCO",
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "c1ccccc1",
        "CC(C)C1=CC=C(C=C1)C(C)C(=O)O",
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
        "[Na+].[Cl-]",
        "C",
    ]
    for smiles in molecules:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        formula, binary = NPClassifier.calculate_fingerprint(mol)
        expected_formula, expected_binary = legacy_fingerprint(smiles)
        assert np.array_equal(formula.reshape(-1), expected_formula), smiles
        assert np.array_equal(binary.reshape(-1), expected_binary), smiles
        assert formula.shape == (1, 2048)
        assert binary.shape == (1, 4096)


def test_batch_fingerprints_and_status_alignment():
    formula, binary, status = NPClassifier.molecules2fingerprints(
        ["CCO", "not-a-smiles", "c1ccccc1"], return_status=True
    )
    assert status == ["ok", "invalid", "ok"]
    assert formula.shape == (3, 2048)
    assert binary.shape == (3, 4096)
    assert np.all(formula[1] == 0)
    assert np.all(binary[1] == 0)
    assert np.any(formula[0] != 0)


def test_on_error_raise():
    try:
        NPClassifier.molecules2fingerprints(["not-a-smiles"], on_error="raise")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid molecules should raise with on_error='raise'")
