from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from util.constant import ELEM_COLOR, VDW_R

from rdkit.Chem import rdDetermineBonds
from rdkit import Chem

from rich import print as rp

def get_color_radii(symbol_s: List[str]) -> Tuple[NDArray, NDArray]:
    colors = [ELEM_COLOR.get(s, (0.5, 0.5, 0.5)) for s in symbol_s]
    radii = np.array([VDW_R.get(s, 1.0) for s in symbol_s])
    return np.array(colors), radii


def build_bond(coor_s: NDArray, symbol_s: List[str]) -> List[Tuple[int, int]]:
    try:
        mol = Chem.RWMol()
        conf = Chem.Conformer(len(symbol_s))
        for i, (sym, crd) in enumerate(zip(symbol_s, coor_s)):
            mol.AddAtom(Chem.Atom(sym))
            conf.SetAtomPosition(i, crd)
        mol.AddConformer(conf)
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        return [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    except Exception as e:
        rp("Error\\[iaw]>: can not gen bond for mol: {}.".format(e))
