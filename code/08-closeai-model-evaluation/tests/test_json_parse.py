from json_parse import extract_prediction


def test_extract_plain_json_prediction():
    content = '{"insurance_decision":{"action":"参保","insurance_type":"城乡居民养老保险"}}'
    pred = extract_prediction(content)
    assert pred.action == "参保"
    assert pred.insurance_type == "城乡居民养老保险"
    assert pred.parse_ok is True


def test_extract_fenced_json_prediction():
    content = '```json\n{"insurance_decision":{"action":"不参保","insurance_type":"不参保"}}\n```'
    pred = extract_prediction(content)
    assert pred.action == "不参保"
    assert pred.insurance_type == "不参保"
    assert pred.parse_ok is True


def test_parse_failure_returns_parse_ok_false():
    pred = extract_prediction("not json")
    assert pred.parse_ok is False
    assert pred.action == "不参保"
    assert pred.insurance_type == "不参保"

