# NAVRIS — Intelligent Navigation & Inertial System

AI-ML based Intelligent Dead Reckoning system for seamless navigation (SIH 2026 Problem Statement 26168).

## Project Structure

`
NAVRIS/
├── data/
│   ├── raw/          # Raw dataset clones/downloads (e.g. IO-VNBD)
│   ├── manifest/     # Structured metadata manifests
│   └── scripts/      # Data parsing, extraction, and validation scripts
├── docs/             # Technical documentation and Phase findings
├── tests/            # Test suite
├── scripts/          # Automation and utility scripts
├── requirements.txt  # Python environment dependencies
└── README.md
`

## Status: Phase 0 (Dataset Ground Truth & Validation)
Currently focused exclusively on empirical inspection of the IO-VNBD dataset, validating sensor columns, sampling rates, coordinate encodings, GPS dropout characteristics, and generating structured recording manifests.
