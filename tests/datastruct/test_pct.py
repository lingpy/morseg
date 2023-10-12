from morseg.datastruct import PCT, PCTNode


def test_init():
    pct = PCT()
    assert isinstance(pct.root, PCTNode)
    assert pct.root.char == ""
    assert len(pct.root.affix_counts) == 0

