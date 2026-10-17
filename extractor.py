import re

def extract_fields(text):
    data = {}

    # Amount
    amount = re.search(r'₹?\s?\d+[.,]?\d*', text)
    if amount:
        data["Amount"] = amount.group()
    else:
        data["Amount"] = "Not Found"

    # Date
    date = re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', text)
    if date:
        data["Date"] = date.group()
    else:
        data["Date"] = "Not Found"

    # Email
    email = re.search(r'\S+@\S+', text)
    if email:
        data["Email"] = email.group()
    else:
        data["Email"] = "Not Found"

    return data