from src.math_operations import add, sub

def test_add():
    assert add(4,5)==9
    assert add(6,2)==8

def test_sub():
    assert sub(6,2)==4
    assert sub(5,8)==-3