import re
import pandas as pd

def extract_money(text):
    """Extract monetary value from string by looking for
    a pattern of a digit, followed by 'euro'.
    e.g. 5 euro --> 5
    Args:
        text (str): Text containing monetary value
    Returns:
        float: The extracted value
    """

    if text:
        extracted_money = re.search("(\d) euro", text).group(1)
        return float(extracted_money)
    else:
        return None