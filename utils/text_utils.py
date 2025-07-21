import re

plate_pattern = re.compile(r'[A-Z]{2}[0-9]{2}[A-Z]{0,2}[0-9]{4}')

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def clean_ocr_results(ocr_results):
    ocr_results.sort(key=lambda r: (min(y for _, y in r[0]), min(x for x, _ in r[0])))
    return [
        ''.join(c for c in r[1].upper().replace("IND", "") if c.isalnum())
        for r in ocr_results
    ]

def match_plate_pattern(text):
    return plate_pattern.search(text)
