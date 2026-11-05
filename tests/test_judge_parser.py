import importlib.util
from pathlib import Path
import sys
import pytest


def parser_module():
    folder=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('judge_recover_test',folder/'judge_recover.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_recovers_only_explicit_grades_from_unescaped_reason_quotes():
    module=parser_module()
    malformed='{"A":{"score":3,"major_error":false,"reason":"O termo "causa" precisa de cuidado."},"B":{"score":4,"major_error":false,"reason":"Correta."}}'
    parsed=module.relaxed_loads(malformed)
    assert parsed['A']['score']==3 and parsed['B']['score']==4
    assert parsed['A']['major_error'] is False
    assert module.last_recovered is True
    with pytest.raises(ValueError):
        module.relaxed_loads(malformed.replace('"score":3','"score":9'))


def test_valid_json_is_preserved_without_recovery():
    module=parser_module()
    assert module.relaxed_loads('{"A":{"score":2}}')=={'A':{'score':2}}
    assert module.last_recovered is False
