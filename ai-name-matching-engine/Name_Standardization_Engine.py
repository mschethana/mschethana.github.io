import csv
from pathlib import Path
from rapidfuzz import process, fuzz

# Automatically detect the folder where this Python script is saved
SCRIPT_DIR = Path(__file__).resolve().parent

# Build full paths to input and output files
file1_path = SCRIPT_DIR / "master_names.csv"
file2_path = SCRIPT_DIR / "input_names.csv"
output_path = SCRIPT_DIR / "standardized_output.csv"


def load_master_data(path):
    """Reads File 1 using built-in csv module (Bypasses NumPy/Pandas)"""
    master_list = []
    try:
        # utf-8-sig handles Excel-generated CSVs and the BOM
        with open(path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 3:
                    # Col A: ID, Col B: First, Col C: Last
                    u_code = row[0].strip()
                    first = row[1].strip()
                    last = row[2].strip()
                    master_list.append({
                        "ID": u_code,
                        "first": first,
                        "last": last,
                        "full": f"{first} {last}"
                    })
    except Exception as e:
        print(f"Error loading master file: {e}")
    return master_list


def match_logic(nickname_full, master_list):
    """
    Returns:
        corrected_name, ID, matching_note, similarity_score
    """
    if not nickname_full or str(nickname_full).strip() == "":
        return None, None, "", None

    nickname_full = str(nickname_full).strip()
    parts = nickname_full.split()

    # 1. Handle single names (no space found)
    if len(parts) < 2:
        all_full_names = [m['full'] for m in master_list]
        res = process.extractOne(
            nickname_full,
            all_full_names,
            scorer=fuzz.token_sort_ratio
        )

        if res and res[1] >= 85:
            matched_name = res[0]
            score = round(res[1], 2)
            ID = next(m['ID'] for m in master_list if m['full'] == matched_name)
            return matched_name, ID, "", score

        return None, None, "", None

    nick_first = parts[0]
    nick_last = parts[-1]

    # 2. Filter for Last Name matches (Case-Insensitive)
    potential_matches = [
        m for m in master_list
        if m['last'].lower() == nick_last.lower()
    ]

    # CASE: Exactly one person found with this last name
    if len(potential_matches) == 1:
        match = potential_matches[0]
        score = fuzz.token_sort_ratio(
            nick_first.lower(),
            match['first'].lower()
        )
        score = round(score, 2)

        if score >= 80:
            return match['full'], match['ID'], "Perfect Match", score
        else:
            return match['full'], match['ID'], "last name matching but might need 2nd review", score

    # CASE: Multiple people with same last name
    if len(potential_matches) > 1:
        # Create a list of first names for the fuzzy processor
        choices = [m['first'] for m in potential_matches]
        result = process.extractOne(
            nick_first,
            choices,
            scorer=fuzz.token_sort_ratio
        )

        if result:
            best_first, score, index = result
            score = round(score, 2)
            match = potential_matches[index]
            note = "Perfect Match" if score >= 80 else "last name matching but might need 2nd review"
            return match['full'], match['ID'], note, score

    # 3. Fallback: Full string match against entire list if no last name match found
    all_full_names = [m['full'] for m in master_list]
    res = process.extractOne(
        nickname_full,
        all_full_names,
        scorer=fuzz.token_sort_ratio
    )

    if res and res[1] >= 85:
        matched_name = res[0]
        score = round(res[1], 2)
        ID = next(m['ID'] for m in master_list if m['full'] == matched_name)
        return matched_name, ID, "Best Match", score

    # No acceptable match found
    return None, None, "No match found", None


# --- MAIN EXECUTION ---

# 1. Load data from the master file
master_records = load_master_data(file1_path)

if master_records:
    # 2. Read input and write directly to output CSV
    with open(file2_path, mode='r', encoding='utf-8-sig') as infile, \
         open(output_path, mode='w', encoding='utf-8', newline='') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write headers for the new file
        next(reader, None)  # Skip input header
        writer.writerow([
            "Nickname",
            "Corrected_Full_Name",
            "ID",
            "Matching_Note",
            "Similarity_Score"
        ])

        # 3. Iterate through names to be matched
        for row in reader:
            if not row:
                continue

            nickname = row[0]

            corrected_name, ID, note, score = match_logic(
                nickname,
                master_records
            )

            writer.writerow([
                nickname,
                corrected_name,
                ID,
                note,
                score
            ])

    print(f"Completed. Output saved to: {output_path}")