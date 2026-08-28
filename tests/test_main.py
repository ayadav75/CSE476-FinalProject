from unittest.mock import patch
from main import validate_answer, AgentStrategies


def test_validate_answer_passthrough():
    assert validate_answer("Paris") == "Paris"


def test_validate_answer_truncates_long_strings():
    long_str = "x" * 6000
    result = validate_answer(long_str)
    assert result.endswith("...[TRUNCATED]")
    assert len(result) < 6000


@patch("main.call_llm")
def test_solve_math_extracts_delimiter_answer(mock_call_llm):
    mock_call_llm.return_value = "0.15 * 80 = 12. #### 12"
    solver = AgentStrategies()
    assert solver.solve_math("What is 15% of 80?") == "12"


@patch("main.call_llm")
def test_solve_math_falls_back_to_answer_line(mock_call_llm):
    mock_call_llm.return_value = "Some reasoning without a delimiter\nAnswer: 17"
    solver = AgentStrategies()
    assert solver.solve_math("Solve for x") == "17"


@patch("main.call_llm")
def test_solve_coding_strips_def_line_from_critique_pass(mock_call_llm):
    mock_call_llm.side_effect = [
        "return x + 1",                  # first-pass draft
        "def foo():\n    return x + 1",  # critique pass wrongly re-adds def
    ]
    solver = AgentStrategies()
    result = solver.solve_coding("Complete the function")
    assert "def " not in result
    assert result.strip() == "return x + 1"


@patch("main.call_llm")
def test_solve_prediction_single_value_boxed(mock_call_llm):
    mock_call_llm.return_value = "\\boxed{Yes}"
    solver = AgentStrategies()
    assert solver.solve_prediction("Will it happen?") == "['Yes']"


@patch("main.call_llm")
def test_solve_prediction_list_boxed(mock_call_llm):
    mock_call_llm.return_value = "\\boxed{['Item1', 'Item2']}"
    solver = AgentStrategies()
    assert solver.solve_prediction("Predict the items") == "['Item1', 'Item2']"


@patch("main.internet_search")
@patch("main.call_llm")
def test_solve_search_based_maps_yes_to_true(mock_call_llm, mock_search):
    mock_search.return_value = ""
    mock_call_llm.return_value = "The search confirms it. #### Yes"
    solver = AgentStrategies()
    assert solver._solve_search_based("Is the sky blue?") == "true"


@patch("main.call_llm")
def test_solve_planning_filters_filler_words(mock_call_llm):
    mock_call_llm.return_value = (
        "Let me think about this...\n"
        "(pick up the block1 from the table)\n"
        "Some extra reasoning line\n"
        "(put down the crate onto the table)"
    )
    solver = AgentStrategies()
    result = solver.solve_planning("Stack block1 on the table")
    assert result == "(pick-up block1 table)\n(put-down onto table)"
