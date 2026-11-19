import pytest
from src.tools import ingredient_converter

def test_ingredient_converter_scales_up():
    result = ingredient_converter.invoke({"quantity": 200, "original_servings": 4, "new_servings": 6})
    assert result == 300.0

def test_ingredient_converter_scales_down():
    result = ingredient_converter.invoke({"quantity": 3, "original_servings": 6, "new_servings": 2})
    assert result == 1.0

@pytest.mark.parametrize("invalid_servings", [0, -2])
def test_ingredient_converter_invalid_servings(invalid_servings):
    with pytest.raises(ValueError, match="greater than zero"):
        ingredient_converter.invoke({"quantity": 100, "original_servings": invalid_servings, "new_servings": 4})

def test_ingredient_converter_negative_quantity():
    with pytest.raises(ValueError, match="cannot be negative"):
        ingredient_converter.invoke({"quantity": -50, "original_servings": 4, "new_servings": 8})