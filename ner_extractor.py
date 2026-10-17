import spacy

nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)

    raw_money = re.findall(r'₹?\s?\d+(?:,\d{3})*(?:\.\d{2})?', text)

    # ✅ FILTER MONEY
    money = []
    for m in raw_money:
        clean = m.replace(",", "").replace("₹", "").strip()

        # Ignore very small or useless numbers
        if len(clean) <= 2:
            continue

        # Ignore pure integers like 01, 3, 7
        if clean.isdigit() and int(clean) < 50:
            continue

        money.append(m)

    return {
        "dates": dates,
        "money": list(set(money)),  # remove duplicates
        "org": [],
        "person": re.findall(r'\S+@\S+', text)
    }