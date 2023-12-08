import pytest
import sys

sys.path.append('./src/')
from preprocessing import *

def test_clean_text_no_special_characters():
    text = "This is a simple text without special characters."
    result = clean_text(text)
    assert result == text.lower()[:-1]

def test_clean_text_with_emojis():
    text = "This is a text with 😀 emojis 🚀."
    result = clean_text(text)
    assert result == text.lower()[:-1]

def test_clean_text_with_special_characters():
    text = "This is a text with @#$ special characters!"
    result = clean_text(text)
    expected_result = "this is a text with special characters"
    assert result == expected_result

def test_clean_text_with_multiple_spaces():
    text = "This     is    a    text    with    multiple    spaces."
    result = clean_text(text)
    expected_result = "this is a text with multiple spaces"
    assert result == expected_result

def test_clean_text_with_special_characters_and_emojis():
    text = "This is a text with @#$ special characters! 😀🚀\n"
    result = clean_text(text)
    expected_result = "this is a text with special characters 😀🚀"
    assert result == expected_result

def test_clean_text_with_repeated_emojis_and_whitespace():
    text = "Text with 😀😀 repeated emojis and extra whitespace. \n 🚀   "
    result = clean_text(text)
    expected_result = "text with 😀😀 repeated emojis and extra whitespace 🚀"
    assert result == expected_result

if __name__ == "__main__":
    pytest.main()