from pathlib import Path
import pytest
from easybfe.mapping import LazyMCSMapper, OpenFEAtomMapper


@pytest.mark.parametrize('method', ['lazymcs', 'lomap', 'kartograf'])
def test_mapping(method):
    ligandA = Path(__file__).parent / 'data/CDD_1819.sdf'
    ligandB = Path(__file__).parent / 'data/CDD_1845.sdf'
    mcs = Path(__file__).parent / 'data/CDD_1845_south_mcs.sdf'
    if method != 'lazymcs':
        mapper = OpenFEAtomMapper(method)
    else:
        mapper = LazyMCSMapper(mcs)
    
    wdir = Path(__file__).parent / f'_test_mcs/{method}'
    wdir.mkdir(exist_ok=True, parents=True)
    mapping = mapper.run(ligandA, ligandB, wdir)
    assert mapping == {
        0: 0, 1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 8, 7: 9, 8: 10, 9: 11, 
        10: 12, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19,
        18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 31, 25: 32,
        26: 33, 27: 34, 28: 35, 29: 36, 30: 37, 31: 38, 32: 39, 33: 40,
        34: 41, 35: 42, 36: 43, 37: 44, 38: 45, 39: 46, 40: 47, 41: 48, 
        42: 49, 43: 50, 44: 51, 45: 52, 46: 53, 47: 54, 48: 55, 49: 56, 
        50: 57, 51: 58, 52: 59, 53: 60, 54: 61
    }
    assert Path.is_file(wdir / 'atom_mapping.json')
    assert Path.is_file(wdir / 'atom_mapping.png')
