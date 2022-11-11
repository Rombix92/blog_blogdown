import sys
sys.path.insert(0, '/Users/lrabalski1/Desktop/prv/blog/content/post/2022-11-11-unit-tests')


from src.regex import extract_money
import pytest


def test_empty_string():
    empty_string = ""
    extracted_money = extract_money(empty_string)
    expected_output = None
    assert extracted_money == expected_output

def test_money_with_decimal_numbers():
    decimal_string = "5.49 euro"
    extracted_money = extract_money(decimal_string)
    assert extracted_money == 5.49