from browserpilot.agent import TOOLS, normalize_key


def test_tools_have_strict_function_schemas() -> None:
    assert TOOLS
    for tool in TOOLS:
        function = tool["function"]
        assert function["parameters"]["additionalProperties"] is False


def test_normalizes_common_model_key_names() -> None:
    assert normalize_key("ENTER") == "Enter"
    assert normalize_key("CTRL+L") == "Control+L"
    assert normalize_key("ESC") == "Escape"
