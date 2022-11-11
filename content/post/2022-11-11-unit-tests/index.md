---
title: Unit Tests
author: Package Build
date: '2022-11-11'
toc: true
categories: ['Unit testing']
tags: ['Python', 'Unit testing']
---



# Problem

What Unit testing are and how they work?

## Introduction:

A unit test aims to check whether a part of your code operates in the intended way. Writing them has the following benefits:

-   Reduces bugs when developing new features or when changing the existing functionality
-   Prevents unexpected output
-   Helps detecting edge cases
-   Tests can serve as documentation

### When should you start writing tests?

-   The more certainty you want to have that your code works as intended, the more you want to invest in testing.

-   Especially when code is used in production, writing tests can be very valuable and ultimately time-saving.

### **When it shouldn't be usefull?**

-   When you are exploring a dataset for the first time, you will be less inclined to write them.

## Python Unit testing

**Pytest** is a great tool for writing unit tests in Python. It makes writing tests short, easy to read, and provides great output. You can install pytest using `pip install pytest` and create a folder structure like the following:

project/\
├─ src/\
│ ├─ regex.py\
│ ├─ **init**.py\
├─ test/\
│ ├─ test_regex.py\
│ ├─ **init**.py

### Use case





In src/regex.py I insert following python code:


```
## Help on function extract_money in module src.regex:
## 
## extract_money(text)
##     Extract monetary value from string by looking for
##     a pattern of a digit, followed by 'euro'.
##     e.g. 5 euro --> 5
##     Args:
##         text (str): Text containing monetary value
##     Returns:
##         float: The extracted value
```

While `test/test_regex.py` I hold following code.


```
## import sys
## sys.path.insert(0, '/Users/lrabalski1/Desktop/prv/blog/content/post/2022-11-11-unit-tests')
## 
## 
## from src.regex import extract_money
## import pytest
## 
## 
## def test_empty_string():
##     empty_string = ""
##     extracted_money = extract_money(empty_string)
##     expected_output = None
##     assert extracted_money == expected_output
## 
## def test_money_with_decimal_numbers():
##     decimal_string = "5.49 euro"
##     extracted_money = extract_money(decimal_string)
##     assert extracted_money == 5.49
```

If the code is localised in the file with suffix "test\_" and within folder "test\_" then running code bellow from `root directory` will run test unit test.


```
## ============================= test session starts ==============================
## platform darwin -- Python 3.8.12, pytest-7.2.0, pluggy-1.0.0 -- /Users/lrabalski1/miniforge3/envs/everyday_use/bin/python3.8
## cachedir: .pytest_cache
## rootdir: /Users/lrabalski1/Desktop/prv/blog/content/post/2022-11-11-unit-tests
## collecting ... collected 2 items
## 
## test/test_regex.py::test_empty_string PASSED                             [ 50%]
## test/test_regex.py::test_money_with_decimal_numbers FAILED               [100%]
## 
## =================================== FAILURES ===================================
## _______________________ test_money_with_decimal_numbers ________________________
## 
##     def test_money_with_decimal_numbers():
##         decimal_string = "5.49 euro"
##         extracted_money = extract_money(decimal_string)
## >       assert extracted_money == 5.49
## E       assert 9.0 == 5.49
## 
## test/test_regex.py:18: AssertionError
## =========================== short test summary info ============================
## FAILED test/test_regex.py::test_money_with_decimal_numbers - assert 9.0 == 5.49
## ========================= 1 failed, 1 passed in 0.33s ==========================
```

As you can see

-   first test went positive, function work well with empty string

-   second test failed, we got an AssertionError, stating that 9 is not equal to 5.49. It seems that our function only extracted the last digit of 5.49. Without this test, we would have been less likely to catch this error. The function would have returned the value 9, instead of throwing us an error.

## Advantages of using Unit Test

The next step is to fix our code and make sure that it now passes the test that failed before, while still passing the first test that we wrote as well. A great thing about unit testing is that it gives you an overview of how your changes affect the project as a whole. The tests will make you aware if any of your code additions cause any unexpected consequences.

Furthermore, the fact that our initial function did not pass the decimal number test raises the question what other examples might come up that we did not anticipate. Maybe we will get a number separated by a comma, like 5,49 euro. Or the monetary value is formatted as € 5.49,-. Unit testing allows us to quickly check for these cases and some other less-common \*\*edge cases\*\*.
