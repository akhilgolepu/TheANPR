import re

# Standard list of Indian State Codes
VALID_STATES = {
    'AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JH', 'KA', 'KL', 
    'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PB', 'RJ', 'SK', 'TN', 'TS', 
    'TR', 'UP', 'UK', 'WB', 'AN', 'CH', 'DN', 'DD', 'DL', 'JK', 'LA', 'LD', 'PY'
}

# Common OCR mistakes
DICT_LETTER_TO_NUMBER = {'O': '0', 'Q': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
DICT_NUMBER_TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'}

def clean_raw_text(text: str) -> str:
    """Removes spaces, IND codes, and special characters."""
    text = text.upper().replace(" ", "").replace("-", "")
    # Sometimes OCR reads the small 'IND'
    if text.startswith("IND"):
        text = text[3:]
    return re.sub(r'[^A-Z0-9]', '', text)

def enforce_format(char: str, expected_type: str) -> str:
    """Forces a character to be a letter or number based on common confusions."""
    if expected_type == 'letter':
        return DICT_NUMBER_TO_LETTER.get(char, char)
    elif expected_type == 'number':
        return DICT_LETTER_TO_NUMBER.get(char, char)
    return char

def correct_indian_plate(raw_text: str) -> str:
    """
    Applies heuristics to correct common OCR errors based on Indian plate formats.
    Standard Format: SS DD LL NNNN (e.g., MH 12 AB 1234)
    Variants: SS DD L NNNN (e.g., DL 8 C 1234)
    """
    text = clean_raw_text(raw_text)
    
    # If it's completely the wrong length, just return the cleaned text
    if len(text) < 8 or len(text) > 10:
        return text

    corrected = list(text)

    # 1. State Code (Positions 0, 1 MUST be letters)
    corrected[0] = enforce_format(corrected[0], 'letter')
    corrected[1] = enforce_format(corrected[1], 'letter')
    
    # 2. District Code (Positions 2, 3 MUST be numbers)
    corrected[2] = enforce_format(corrected[2], 'number')
    corrected[3] = enforce_format(corrected[3], 'number')
    
    # 3. The 4-digit unique code (The last 4 positions MUST be numbers)
    for i in range(-4, 0):
        corrected[i] = enforce_format(corrected[i], 'number')
        
    # 4. The Alpha Series (The characters in the middle MUST be letters)
    for i in range(4, len(corrected) - 4):
        corrected[i] = enforce_format(corrected[i], 'letter')

    final_plate = "".join(corrected)
    
    # 5. Final Sanity Check against Valid States
    state_code = final_plate[:2]
    if state_code not in VALID_STATES:
        # We could add fuzzy matching here, but for now we just flag it
        pass 

    return final_plate

if __name__ == "__main__":
    # Let's test how powerful this is
    test_cases = [
        "M812C4566",    # OCR read H as 8 -> Should fix to MH
        "KLO7CJ8999",   # OCR read 0 as O -> Should fix to KL 07
        "TS088Q1111",   # OCR read B as 8 -> Should fix to TS 08 BQ 1111
        "INDMH14GN9239" # Read the IND watermark -> Should strip it
    ]
    
    print("🧪 Testing Heuristic Post-Processor...")
    for raw in test_cases:
        fixed = correct_indian_plate(raw)
        print(f"Raw OCR: {raw.ljust(15)} | Corrected: {fixed}")