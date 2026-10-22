# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Restructured documentation into numbered `docs/{en,zh-CN}/` pages with a docs
  hub, language indexes and Level/Prerequisites/Next metadata.
- Added `AGENTS.md`, `AUTHORS`, `NOTICE`, `CITATION.cff` and `Makefile`.
- README is now English-first with a Chinese section; the NPClassifier paper
  citation is included in README and `CITATION.cff`.
- Removed the maintainer metadata; the package lists David Niteas as author.
- Package metadata now points to the LICENSE file and includes the new release
  files in the sdist.

## [0.3.1] - 2026-10-22

### Changed

- Set the wrapper author to David Niteas and update the license copyright
  notice. Upstream attribution and licensing are unchanged.

## [0.3.0] - 2026-10-21

### Added

- `predict_official()`: a direct port of the official NPClassifier decision
  logic (class 0.1, superclass 0.3, pathway 0.5, hierarchical voting,
  glycoside SMARTS check).
- `npc predict`, `npc convert`, `npc verify` and `npc doctor` subcommands. The
  historical single-command `npc --model-dir ...` form remains supported.
- `npc --mode official` for table input; `scores` remains the default mode.
- Bilingual CLI and references documentation.

### Notes

- `predict_scores` / `batch_predict` remains the raw per-class probability API.
- Official thresholds are decision rules, not properties of the sigmoid output;
  the model itself is trained with binary cross-entropy.
- Exact result lists depend on the model release (V1 ONNX vs the official V25
  deployment).

## [0.2.0] - 2026-10-21

### Added

- ID-aligned/scores-mode hardening, HDF5-to-ONNX converter, parity verifier,
  bilingual docs and synthetic tests.
