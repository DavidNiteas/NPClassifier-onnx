"""ONNX inference for NPClassifier.

This module provides the model-loading, fingerprinting and decoding code used by
the Python API and the ``npc`` command line tool.  Model weights are **not**
bundled with this package; see ``docs/conversion.md`` for the supported
conversion workflow.
"""

from __future__ import annotations

import json
import numbers
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm

MODEL_NAMES: Tuple[str, ...] = ("class", "pathway", "superclass")
DEFAULT_MODEL_SET: Tuple[str, ...] = MODEL_NAMES
FINGERPRINT_SIZE = 2048
FINGERPRINT_RADIUS = 2

MoleculeInput = Union[str, Mol]


class NPClassifier:
    """Run NPClassifier inference with ONNX Runtime.

    Parameters
    ----------
    model_dir_path:
        Directory containing ``class_model.onnx``, ``pathway_model.onnx``,
        ``superclass_model.onnx`` (or ``<model>.onnx``) and an ``index*.json``
        label mapping.  Only the requested models have to be present.
    model_set:
        Models to load.  Defaults to all three models.
    """

    def __init__(
        self,
        model_dir_path: Union[str, Path],
        model_set: Optional[Sequence[str]] = None,
    ) -> None:
        self.model_dir_path = Path(model_dir_path).expanduser()
        self.sessions, self.mapping = self.load_sessions(
            self.model_dir_path, model_set
        )

    # ------------------------------------------------------------------
    # Molecule parsing and fingerprinting
    # ------------------------------------------------------------------
    @staticmethod
    def get_mol(inchi_or_smiles: str) -> Mol:
        """Parse a SMILES or InChI string and add explicit hydrogens."""
        if not isinstance(inchi_or_smiles, str):
            raise TypeError("molecule input must be a SMILES/InChI string")
        value = inchi_or_smiles.strip()
        if not value:
            raise ValueError("empty molecule string")
        if value.startswith("InChI"):
            mol = Chem.MolFromInchi(value)
            kind = "InChI"
        else:
            mol = Chem.MolFromSmiles(value)
            kind = "SMILES"
        if mol is None:
            raise ValueError("could not parse %s: %r" % (kind, value))
        return Chem.AddHs(mol)

    @staticmethod
    def calculate_fingerprint(
        mol: Mol,
        n_bits: int = FINGERPRINT_SIZE,
        radius: int = FINGERPRINT_RADIUS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the NPClassifier formula/count fingerprints.

        The historical implementation used
        ``GetMorganFingerprintAsBitVect`` and then counted, for every bit, the
        atom environments at each radius.  The same values are obtained more
        directly with Morgan count fingerprints:

        * ``formula`` is the radius-0 count vector;
        * the second vector contains the exact radius-1 and radius-2 counts.

        The result is numerically identical to the original implementation,
        including duplicated atom environments.
        """
        if not isinstance(mol, Mol):
            raise TypeError("mol must be an rdkit.Chem.Mol instance")
        if n_bits < 1:
            raise ValueError("n_bits must be positive")
        if radius < 0:
            raise ValueError("radius must be non-negative")

        counts: List[np.ndarray] = []
        for current_radius in range(radius + 1):
            generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=current_radius, fpSize=n_bits
            )
            fingerprint = generator.GetCountFingerprint(mol)
            values = np.zeros(n_bits, dtype=np.float32)
            for index, count in fingerprint.GetNonzeroElements().items():
                values[int(index)] = float(count)
            counts.append(values)

        formula = counts[0].reshape(1, n_bits)
        if radius == 0:
            binary = np.zeros((1, 0), dtype=np.float32)
        else:
            exact_radius = [
                counts[current] - counts[current - 1]
                for current in range(1, radius + 1)
            ]
            binary = np.concatenate(exact_radius).reshape(1, n_bits * radius)
        return formula.astype(np.float32), binary.astype(np.float32)

    @staticmethod
    def molecules2fingerprints(
        input_list: Sequence[MoleculeInput],
        *,
        return_status: bool = False,
        on_error: str = "zero",
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, List[str]],
    ]:
        """Featurize a sequence of molecules.

        Invalid molecules produce zero fingerprints and a ``"invalid"`` status
        by default, so that output rows stay aligned with the input.  Set
        ``on_error="raise"`` to fail on the first invalid molecule instead.

        Returns ``(formula, binary)`` or, with ``return_status=True``,
        ``(formula, binary, status)`` where each status is ``"ok"`` or
        ``"invalid"``.
        """
        if on_error not in ("zero", "raise"):
            raise ValueError("on_error must be 'zero' or 'raise'")
        if isinstance(input_list, (str, Mol)):
            input_list = [input_list]

        formulas: List[np.ndarray] = []
        binaries: List[np.ndarray] = []
        status: List[str] = []

        for item in input_list:
            try:
                if isinstance(item, str):
                    mol = NPClassifier.get_mol(item)
                elif isinstance(item, Mol):
                    mol = Chem.AddHs(item)
                else:
                    raise TypeError(
                        "input items must be SMILES/InChI strings or rdkit Mol objects"
                    )
                formula, binary = NPClassifier.calculate_fingerprint(mol)
                status.append("ok")
            except Exception:
                if on_error == "raise":
                    raise
                formula = np.zeros((1, FINGERPRINT_SIZE), dtype=np.float32)
                binary = np.zeros(
                    (1, FINGERPRINT_SIZE * FINGERPRINT_RADIUS), dtype=np.float32
                )
                status.append("invalid")
            formulas.append(formula)
            binaries.append(binary)

        if formulas:
            formula_array = np.vstack(formulas).astype(np.float32, copy=False)
            binary_array = np.vstack(binaries).astype(np.float32, copy=False)
        else:
            formula_array = np.zeros((0, FINGERPRINT_SIZE), dtype=np.float32)
            binary_array = np.zeros(
                (0, FINGERPRINT_SIZE * FINGERPRINT_RADIUS), dtype=np.float32
            )

        if return_status:
            return formula_array, binary_array, status
        return formula_array, binary_array

    # ------------------------------------------------------------------
    # Label mapping and model loading
    # ------------------------------------------------------------------
    @staticmethod
    def load_mapping(mapping_path: Union[str, Path]) -> Dict[str, pd.Series]:
        """Load an NPClassifier ``index*.json`` label mapping.

        The official file is a dictionary of ``{label: index}`` per model
        (``Class``, ``Pathway``, ``Superclass``) plus optional hierarchy
        entries.  Hierarchy entries are ignored.
        """
        path = Path(mapping_path)
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("label mapping must be a JSON object")

        mapping: Dict[str, pd.Series] = {}
        for raw_name, labels in raw.items():
            name = str(raw_name).lower()
            if name not in MODEL_NAMES:
                continue
            if not isinstance(labels, dict):
                raise ValueError(
                    "label mapping entry %r must map labels to indices" % raw_name
                )
            pairs = []
            for label, index in labels.items():
                try:
                    numeric_index = int(index)
                except (TypeError, ValueError):
                    raise ValueError(
                        "label mapping index for %r is not an integer: %r"
                        % (label, index)
                    ) from None
                pairs.append((numeric_index, str(label)))
            pairs.sort(key=lambda item: item[0])
            indices = [item[0] for item in pairs]
            if len(set(indices)) != len(indices):
                raise ValueError(
                    "label mapping entry %r contains duplicate indices" % raw_name
                )
            mapping[name] = pd.Series(
                [item[1] for item in pairs], index=indices, dtype=object
            )

        if not mapping:
            raise ValueError(
                "label mapping has no Class/Pathway/Superclass entries: %s" % path
            )
        return mapping

    @staticmethod
    def _mapping_key(model_name: str) -> str:
        name = str(model_name).lower()
        # Check "superclass" first: it contains the substring "class".
        for key in ("superclass", "pathway", "class"):
            if name == key or name.startswith(key + "_") or name.startswith(key):
                return key
        raise ValueError("unknown NPClassifier model name: %r" % model_name)

    @staticmethod
    def _resolve_model_file(model_dir: Path, model_name: str) -> Path:
        direct = [
            model_dir / ("%s_model.onnx" % model_name),
            model_dir / ("%s.onnx" % model_name),
        ]
        for candidate in direct:
            if candidate.is_file():
                return candidate

        matches = [
            path
            for path in sorted(model_dir.glob("*.onnx"))
            if path.stem == model_name or path.stem.startswith(model_name + "_")
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                "multiple ONNX files match model %r: %s"
                % (model_name, ", ".join(path.name for path in matches))
            )
        available = ", ".join(path.name for path in sorted(model_dir.glob("*.onnx")))
        raise FileNotFoundError(
            "missing %s model in %s (expected %s_model.onnx; found: %s)"
            % (model_name, model_dir, model_name, available or "none")
        )

    @staticmethod
    def _resolve_index_file(model_dir: Path) -> Path:
        for name in ("index.json", "index_v1.json"):
            candidate = model_dir / name
            if candidate.is_file():
                return candidate
        matches = sorted(model_dir.glob("index*.json"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                "multiple label mapping files found: %s; keep only one index*.json"
                % ", ".join(path.name for path in matches)
            )
        raise FileNotFoundError(
            "missing label mapping in %s (expected index.json or index_v1.json)"
            % model_dir
        )

    @staticmethod
    def get_inputs(session: ort.InferenceSession) -> List[str]:
        """Return input names in ONNX Runtime order."""
        return [input_meta.name for input_meta in session.get_inputs()]

    @staticmethod
    def _input_map(session: ort.InferenceSession) -> Dict[str, str]:
        """Map ``formula``/``binary`` to the session's actual input names.

        Existing NPClassifier exports use ``inputs``/``inputs_1``; exports made
        by the bundled converter use ``input_2048``/``input_4096``.  Mapping by
        the last dimension supports both without relying on names or order.
        """
        found: Dict[str, str] = {}
        for input_meta in session.get_inputs():
            shape = input_meta.shape
            if not shape:
                raise ValueError(
                    "model input %r has no shape information" % input_meta.name
                )
            feature_dim = shape[-1]
            if not isinstance(feature_dim, numbers.Integral):
                raise ValueError(
                    "model input %r has a dynamic feature dimension %r; "
                    "expected a fixed 2048 or 4096"
                    % (input_meta.name, shape)
                )
            feature_dim = int(feature_dim)
            if feature_dim == FINGERPRINT_SIZE:
                key = "formula"
            elif feature_dim == FINGERPRINT_SIZE * FINGERPRINT_RADIUS:
                key = "binary"
            else:
                raise ValueError(
                    "model input %r has unsupported feature dimension %d; "
                    "expected 2048 or 4096"
                    % (input_meta.name, feature_dim)
                )
            if key in found:
                raise ValueError(
                    "model has multiple %s-sized inputs; expected exactly one "
                    "2048-wide and one 4096-wide input" % feature_dim
                )
            found[key] = input_meta.name

        missing = {"formula", "binary"} - set(found)
        if missing:
            raise ValueError(
                "model is missing a %s input; inputs are mapped by their last "
                "dimension (2048 and 4096)"
                % " and ".join(sorted(missing))
            )
        return found

    @staticmethod
    def load_sessions(
        model_dir_path: Union[str, Path],
        model_set: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, ort.InferenceSession], Dict[str, pd.Series]]:
        """Load ONNX sessions and the label mapping.

        Returned session keys are the ONNX file stems (for example
        ``class_model``), which are also used as output column prefixes.
        """
        model_dir = Path(model_dir_path).expanduser()
        if not model_dir.exists():
            raise FileNotFoundError("model directory does not exist: %s" % model_dir)
        if not model_dir.is_dir():
            raise NotADirectoryError("model path is not a directory: %s" % model_dir)

        if model_set is None:
            models = list(DEFAULT_MODEL_SET)
        elif isinstance(model_set, str):
            models = [model_set]
        else:
            models = list(model_set)
        normalised: List[str] = []
        for model in models:
            name = str(model).strip().lower()
            if name not in MODEL_NAMES:
                raise ValueError(
                    "unknown model %r; choose from %s"
                    % (model, ", ".join(MODEL_NAMES))
                )
            if name not in normalised:
                normalised.append(name)
        if not normalised:
            raise ValueError("model_set must contain at least one model")

        mapping_path = NPClassifier._resolve_index_file(model_dir)
        mapping = NPClassifier.load_mapping(mapping_path)
        missing_mappings = [name for name in normalised if name not in mapping]
        if missing_mappings:
            raise ValueError(
                "label mapping %s has no entry for: %s"
                % (mapping_path.name, ", ".join(missing_mappings))
            )

        sessions: Dict[str, ort.InferenceSession] = {}
        try:
            for model_name in normalised:
                model_path = NPClassifier._resolve_model_file(model_dir, model_name)
                session = ort.InferenceSession(str(model_path))
                input_map = NPClassifier._input_map(session)
                outputs = session.get_outputs()
                if len(outputs) != 1:
                    raise ValueError(
                        "model %s must have exactly one output, found %d"
                        % (model_path.name, len(outputs))
                    )
                output_shape = outputs[0].shape
                if len(output_shape) != 2:
                    raise ValueError(
                        "model %s output must be 2-D (batch, classes), got %r"
                        % (model_path.name, output_shape)
                    )
                output_dim = output_shape[-1]
                expected_classes = len(mapping[model_name])
                if isinstance(output_dim, numbers.Integral) and int(output_dim) != expected_classes:
                    raise ValueError(
                        "model %s outputs %d classes but the label mapping has %d; "
                        "check that the ONNX model and index file belong together"
                        % (model_path.name, int(output_dim), expected_classes)
                    )

                session_key = model_path.stem
                if session_key in sessions:
                    raise ValueError("duplicate model file stem: %s" % session_key)
                sessions[session_key] = session
        except Exception:
            for session in sessions.values():
                # ONNX Runtime sessions are released by reference counting.
                del session
            raise

        return sessions, mapping

    # ------------------------------------------------------------------
    # Prediction and decoding
    # ------------------------------------------------------------------
    def batch_predict(
        self,
        input_list: Sequence[MoleculeInput],
        used_model: Union[Sequence[str], str] = "auto",
        *,
        return_status: bool = False,
    ) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], List[str]]]:
        """Predict class probabilities for a batch of molecules.

        Output arrays always have one row per input, in input order.  Rows for
        invalid molecules are all zeros; use ``return_status=True`` to obtain
        the per-row ``"ok"``/``"invalid"`` status.
        """
        if isinstance(input_list, (str, Mol)):
            input_list = [input_list]
        items = list(input_list)
        formula, binary, status = self.molecules2fingerprints(
            items, return_status=True
        )

        if used_model == "auto":
            selected_keys = list(self.sessions)
        elif isinstance(used_model, str):
            allowed = {used_model.lower()}
            selected_keys = [
                key
                for key in self.sessions
                if key.lower() in allowed or self._mapping_key(key) in allowed
            ]
        else:
            allowed = {str(name).lower() for name in used_model}
            selected_keys = [
                key
                for key in self.sessions
                if key.lower() in allowed or self._mapping_key(key) in allowed
            ]

        results: Dict[str, np.ndarray] = {}
        for session_key in selected_keys:
            session = self.sessions[session_key]
            inputs = self._input_map(session)
            if items:
                result = session.run(
                    None,
                    {
                        inputs["formula"]: formula,
                        inputs["binary"]: binary,
                    },
                )[0]
                result = np.asarray(result, dtype=np.float32).copy()
                if result.ndim != 2 or result.shape[0] != len(items):
                    raise ValueError(
                        "model %s returned shape %r for %d inputs; expected "
                        "(batch, classes)"
                        % (session_key, result.shape, len(items))
                    )
            else:
                classes = len(self.mapping[self._mapping_key(session_key)])
                result = np.zeros((0, classes), dtype=np.float32)

            for row, row_status in enumerate(status):
                if row_status != "ok":
                    result[row, :] = 0.0
            results[session_key] = result

        if return_status:
            return results, status
        return results

    def index2label(
        self,
        index: Union[int, Sequence[int], np.ndarray],
        model_name: str,
    ) -> Union[str, List[str]]:
        """Translate one index or a sequence of indices to labels."""
        mapping_key = self._mapping_key(model_name)
        series = self.mapping[mapping_key]

        if isinstance(index, numbers.Integral):
            position = int(index)
            if position < 0 or position >= len(series):
                raise IndexError(
                    "label index %d out of range for %s (%d labels)"
                    % (position, mapping_key, len(series))
                )
            return str(series.iloc[position])

        indices = np.asarray(index)
        if indices.ndim == 0:
            position = int(indices)
            if position < 0 or position >= len(series):
                raise IndexError(
                    "label index %d out of range for %s (%d labels)"
                    % (position, mapping_key, len(series))
                )
            return str(series.iloc[position])
        if indices.ndim != 1:
            raise ValueError("index must be an integer or a one-dimensional sequence")

        positions = []
        for value in indices:
            position = int(value)
            if position < 0 or position >= len(series):
                raise IndexError(
                    "label index %d out of range for %s (%d labels)"
                    % (position, mapping_key, len(series))
                )
            positions.append(position)
        return [str(label) for label in series.iloc[positions].tolist()]

    def indexs2labels(
        self,
        indexs: np.ndarray,
        model_name: str,
    ) -> np.ndarray:
        """Translate an array of indices, preserving its shape."""
        array = np.asarray(indexs)
        shape = array.shape
        labels = self.index2label(array.reshape(-1), model_name)
        return np.asarray(labels, dtype=object).reshape(shape)

    @staticmethod
    def get_top_k_index(array: np.ndarray, top_k: int) -> np.ndarray:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        values = np.asarray(array).reshape(-1)
        if top_k > values.size:
            raise ValueError(
                "top_k=%d is larger than the number of classes (%d)"
                % (top_k, values.size)
            )
        return np.argsort(values)[-top_k:][::-1].astype(int)

    @staticmethod
    def get_top_k_prob(array: np.ndarray, top_k: int) -> np.ndarray:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        values = np.asarray(array).reshape(-1)
        if top_k > values.size:
            raise ValueError(
                "top_k=%d is larger than the number of classes (%d)"
                % (top_k, values.size)
            )
        return np.sort(values)[-top_k:][::-1]

    def decode_result(
        self,
        results: np.ndarray,
        model_name: str,
        top_k: int = 1,
        valid: Optional[Sequence[bool]] = None,
    ) -> pd.DataFrame:
        """Decode one model's probability matrix, keeping one row per input."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        array = np.asarray(results)
        if array.ndim != 2:
            raise ValueError("results must be a 2-D array")
        if valid is None:
            valid = [bool(np.sum(row) > 0) for row in array]
        if len(valid) != len(array):
            raise ValueError("valid mask length does not match the number of rows")

        columns: List[str] = []
        for rank in range(1, top_k + 1):
            columns.append("%s_prob_%d" % (model_name, rank))
            columns.append("%s_label_%d" % (model_name, rank))

        rows: List[Dict[str, object]] = []
        for row_index, row in enumerate(array):
            decoded: Dict[str, object] = {}
            if valid[row_index]:
                indices = self.get_top_k_index(row, top_k)
                probabilities = self.get_top_k_prob(row, top_k)
                labels = self.index2label(indices, model_name)
                for rank in range(top_k):
                    decoded["%s_prob_%d" % (model_name, rank + 1)] = probabilities[rank]
                    decoded["%s_label_%d" % (model_name, rank + 1)] = labels[rank]
            rows.append(decoded)
        return pd.DataFrame(rows, columns=columns, index=range(len(rows)))

    def decode_results(
        self,
        results_dict: Dict[str, np.ndarray],
        top_k: int = 1,
        status: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Decode all models and add a per-row ``status`` column."""
        if status is not None:
            status = list(status)
        if not results_dict:
            frame = pd.DataFrame(index=range(len(status) if status is not None else 0))
            if status is not None:
                frame.insert(0, "status", status)
            return frame

        frames = []
        for model_name, results in results_dict.items():
            valid = None
            if status is not None:
                if len(status) != len(results):
                    raise ValueError(
                        "status length does not match model output rows"
                    )
                valid = [item == "ok" for item in status]
            frames.append(
                self.decode_result(results, model_name, top_k=top_k, valid=valid)
            )
        frame = pd.concat(frames, axis=1)
        if status is not None:
            frame.insert(0, "status", status)
        return frame

    def batch_predict_and_decode(
        self,
        input_list: Sequence[MoleculeInput],
        used_model: Union[Sequence[str], str] = "auto",
        top_k: int = 1,
    ) -> pd.DataFrame:
        """Predict and decode in one call; invalid rows remain aligned."""
        results, status = self.batch_predict(
            input_list, used_model=used_model, return_status=True
        )
        return self.decode_results(results, top_k=top_k, status=status)

    # ------------------------------------------------------------------
    # Table input
    # ------------------------------------------------------------------
    @staticmethod
    def _read_table_column(
        data_path: Path,
        tag_column: str,
    ) -> Sequence[MoleculeInput]:
        suffix = data_path.suffix.lower()
        try:
            if suffix == ".csv":
                frame = pd.read_csv(data_path)
            elif suffix in (".tsv", ".tab", ".txt"):
                frame = pd.read_table(data_path, sep="\t")
            elif suffix in (".xlsx", ".xls"):
                frame = pd.read_excel(data_path)
            elif suffix == ".pkl":
                frame = pd.read_pickle(data_path)
            elif suffix == ".parquet":
                frame = pd.read_parquet(data_path)
            elif suffix in (".h5", ".hdf5", ".hdf"):
                try:
                    frame = pd.read_hdf(data_path)
                except ImportError as exc:
                    raise ImportError(
                        "reading HDF5 input requires the optional 'tables' "
                        "dependency; install with: pip install 'NPClassifier_onnx[hdf]'"
                    ) from exc
            else:
                raise ValueError(
                    "unsupported input file extension %r; use csv, tsv/txt, "
                    "xlsx, pkl, parquet or h5" % suffix
                )
        except FileNotFoundError:
            raise
        except KeyError:
            raise

        if tag_column not in frame.columns:
            lower_columns = {str(column).lower(): column for column in frame.columns}
            actual_column = lower_columns.get(str(tag_column).lower())
            if actual_column is None and str(tag_column).lower() in ("smiles", "inchi"):
                for alternative in ("smiles", "inchi"):
                    if alternative in lower_columns:
                        actual_column = lower_columns[alternative]
                        break
            if actual_column is None:
                raise KeyError(
                    "column %r not found in %s; available columns: %s"
                    % (tag_column, data_path, ", ".join(map(str, frame.columns)))
                )
            tag_column = actual_column
        return frame[tag_column].tolist()

    def predict_from_table_file(
        self,
        data_path: Union[str, Path],
        tag_column: str = "smiles",
        batch_size: int = 256,
        top_k: int = 1,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Predict molecules from a table column.

        The returned frame has one row per input molecule and is indexed by the
        original molecule strings.  Invalid molecules are reported with
        ``status == "invalid"`` and empty prediction columns.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        path = Path(data_path).expanduser()
        data = list(self._read_table_column(path, tag_column))
        if not data:
            raise ValueError("input table has no rows: %s" % path)

        batches = [
            data[start : start + batch_size]
            for start in range(0, len(data), batch_size)
        ]
        frames = []
        for batch in tqdm(batches, disable=not progress):
            frames.append(self.batch_predict_and_decode(batch, top_k=top_k))
        frame = pd.concat(frames, ignore_index=True)
        frame.index = pd.Index(data, name=tag_column)
        return frame
