def test_data_contains_correct_columns(data):
    """Test that the input data contains expected columns."""
    expected_columns = [
        'age',
        'workclass',
        'fnlgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'salary'
    ]

    assert list(data.columns) == expected_columns
