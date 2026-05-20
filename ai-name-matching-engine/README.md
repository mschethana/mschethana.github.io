# AI-Assisted Name Matching Engine

A Python-based fuzzy matching engine that standardizes inconsistent names (nicknames, abbreviations, reordered names, and partial names) against an official master list.

This project uses AI-assisted fuzzy logic matching with custom decision rules to automatically identify the most likely official name and associated ID.

---

## Business Problem

In enterprise systems, names are often entered in different formats:

- Mike Thomas
- Michael Thomas
- Thomas Michael
- M Thomas
- Joe Mathew
- Joseph Mathew

These inconsistencies create challenges in:

- Survey assignments
- Data validation
- Reporting
- User mapping
- Dashboard filtering
- Work Allocation/ Tracking

This project solves that problem by automatically matching non-standard names to official master data.

---

## Solution Overview

The script performs a multi-step matching process:

1. Loads official names and IDs from a master CSV file.
2. Identifies potential candidates using last-name matching and creates a shortlist of all matched candidates
3. Uses fuzzy similarity scoring to compare first names for this shortlist of candidates.
4. Applies confidence thresholds to determine match quality.
5. In case, last name match is not found, uses a fallback fuzzy search across the full name list, ignoring the order.
6. Outputs the nickname, corrected name, ID, matching notes, and similarity score in csv format.

---

## Key Features

- Dynamic path resolution using `Path(__file__).resolve().parent`
- No dependency on Pandas or NumPy
- RapidFuzz-based fuzzy matching
- Confidence-based similarity scores
- Custom business rules
- Review notes for lower-confidence matches
- CSV input and output
- Reusable across different datasets

---

## Technologies Used

- Python
- RapidFuzz
- CSV
- Pathlib
- Fuzzy String Matching
- Entity Resolution
- Probabilistic Record Linkage

---

## Statistical Concepts Applied

This project demonstrates several data science concepts:

- Similarity scoring
- Probabilistic matching
- Confidence thresholds
- Candidate reduction
- Rule-based decision systems
- Entity resolution

The similarity score serves as a confidence measure indicating the likelihood of a proposed match being correct.

---

## Project Structure

```text
ai-name-matching-engine/
├── surveyor_name_standardization.py
├── master_names.csv
├── input_names.csv
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
