import pytest
from unittest.mock import patch

MOCK_DATASET = {
    "test": [{
        "question":
        "What is the Answer to the Great Question of Life, The Universe, and Everything?",
        "answer": "#### 42"
    }, {
        "question": "Vegeta, what does the scouter say about his power level?",
        "answer": "IT'S OVER #### 9000"
    }],
    "train": [{
        "question": "What number indicates that someone is a powerful hacker?",
        "answer": "That can only refer to one thing: they are elite, aka #### 1337"
    }]
}

@pytest.fixture()
def patch_load_dataset():
    with patch("evaluator.datasets.load_dataset", return_value=MOCK_DATASET):
        yield
