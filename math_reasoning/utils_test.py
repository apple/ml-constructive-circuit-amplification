# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

from gsm_utils import extract_until_first_answer


def test_extract_until_first_answer_single():
    """Test extraction when there's only one answer sentence."""
    text = "Let me solve this step by step. First, I need to calculate 2 + 3. The answer is 5."
    expected = "Let me solve this step by step. First, I need to calculate 2 + 3. The answer is 5."
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_multiple():
    """Test extraction stops at the first answer sentence when multiple exist."""
    text = "Let me solve this step by step. The answer is 5. But wait, let me check again. The answer is actually 7."
    expected = "Let me solve this step by step. The answer is 5."
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_with_additional_text():
    """Test extraction with answer sentence containing additional text."""
    text = "Let me solve this step by step. The answer is 5 because 2 + 3 = 5. Let me verify this."
    expected = "Let me solve this step by step. The answer is 5 because 2 + 3 = 5."
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_at_beginning():
    """Test extraction when answer sentence is at the very beginning."""
    text = "The answer is 5. Let me explain how I got this."
    expected = "The answer is 5."
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_at_end():
    """Test extraction when answer sentence is at the very end."""
    text = "Let me solve this step by step. First, I need to calculate 2 + 3. The answer is 5."
    expected = "Let me solve this step by step. First, I need to calculate 2 + 3. The answer is 5."
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_no_match():
    """Test extraction when no answer sentence is found."""
    text = "Let me solve this step by step. First, I need to calculate 2 + 3. This gives us 5."
    expected = "Let me solve this step by step. First, I need to calculate 2 + 3. This gives us 5."
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_empty():
    """Test extraction with empty string."""
    text = ""
    expected = ""
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_whitespace_only():
    """Test extraction with whitespace-only string."""
    text = "   \n\t   "
    expected = "   \n\t   "
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_without_period():
    """Test extraction when answer sentence doesn't end with a period."""
    text = "Let me solve this step by step. The answer is 5"
    expected = "Let me solve this step by step. The answer is 5"
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_multiple_periods():
    """Test extraction with answer sentence containing multiple periods."""
    text = "Let me solve this step by step. The answer is 5.0. Let me verify this."
    expected = "Let me solve this step by step. The answer is 5."
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_case_sensitive():
    """Test extraction is case sensitive - only matches 'answer is' not 'ANSWER IS'."""
    text = "Let me solve this step by step. The ANSWER IS 5. Let me verify this."
    expected = "Let me solve this step by step. The ANSWER IS 5. Let me verify this."
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_numbers_symbols():
    """Test extraction with answer containing numbers and mathematical symbols."""
    text = "Let me solve this step by step. The answer is 2.5 + 3.7 = 6.2. Let me verify this."
    expected = "Let me solve this step by step. The answer is 2."
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_special_chars():
    """Test extraction with answer containing special characters."""
    text = "Let me solve this step by step. The answer is $5.99. Let me verify this."
    expected = "Let me solve this step by step. The answer is $5."
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_stripping():
    """Test that the result is properly stripped of leading/trailing whitespace."""
    text = "  Let me solve this step by step. The answer is 5.  "
    expected = "Let me solve this step by step. The answer is 5."
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_newlines():
    """Test extraction with answer sentence containing newlines."""
    text = "Let me solve this step by step.\nThe answer is 5.\nLet me verify this."
    expected = "Let me solve this step by step.\nThe answer is 5."
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_tabs():
    """Test extraction with answer sentence containing tabs."""
    text = "Let me solve this step by step.\tThe answer is 5.\tLet me verify this."
    expected = "Let me solve this step by step.\tThe answer is 5."
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_unicode():
    """Test extraction with answer containing unicode characters."""
    text = "Let me solve this step by step. The answer is 5°C. Let me verify this."
    expected = "Let me solve this step by step. The answer is 5°C."
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_quotes():
    """Test extraction with answer containing quotes."""
    text = "Let me solve this step by step. The answer is '5'. Let me verify this."
    expected = "Let me solve this step by step. The answer is '5'."
    result = extract_until_first_answer(text)
    assert result == expected


def test_extract_until_first_answer_parentheses():
    """Test extraction with answer containing parentheses."""
    text = "Let me solve this step by step. The answer is (5). Let me verify this."
    expected = "Let me solve this step by step. The answer is (5)."
    result = extract_until_first_answer(text)
    assert result == expected
