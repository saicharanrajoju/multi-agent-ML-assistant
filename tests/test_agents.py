# Check if agents can be imported
from src.agents import profiler, cleaner, feature_eng, modeler, critic

def test_imports():
    assert profiler
    assert cleaner
    assert feature_eng
    assert modeler
    assert critic
