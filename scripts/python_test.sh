pytest tests/ -q --ignore=tests/test_quant.py --ignore=tests/test_samplers.py --ignore=tests/test_screening.py
pytest tests/test_quant.py tests/test_samplers.py tests/test_screening.py -q 2>&1 | tail -n 5 || true
