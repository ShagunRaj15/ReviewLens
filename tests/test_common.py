import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import clean_text

def test_clean_text_removes_html_and_normalizes_spaces():
    assert clean_text("Good<br />  product <b>quality</b>") == "Good product quality"
