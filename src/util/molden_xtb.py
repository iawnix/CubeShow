import numpy as np
from types import MappingProxyType
from typing import Any, List, Callable, Dict
import sys
import os
from rich import print as rprint
from rich.table import Table
from rich.text import Text


class basis_gaussian_args():
    def __init__(self):
        self.exponential_factor: float = None
        self.contraction_coefficient: float = None

class basis_args():

    SHELL_MULTI: Dict[str, int] = {
              "s":1
             ,"p":3
             ,"d":5
            }

    # 严禁修改
    SHELL_MULTI = MappingProxyType(SHELL_MULTI)

    def __init__(self) -> None:
        self.shell_label: str = None
        self.number_of_primitives: int = None
        self.scaling_factor_of_basis_function: float = None
        self.basis_gaussian_args_s: List[Any] = []

class GTO():
    """
    GTO: Gaussian Type Orbitals
    """
    def __init__(self) -> None:
        self.atom_sequence_number: int = None
        self.angular_quantum_number: int = None
        self.basis_args: List[Any] = []


class mo_coefficient():

    def __init__(self) -> None:

        self.ao_number_index: int = None
        self.mo_coefficient: float = None


class MO():
    """
    MO: molecular orbitals
    """
    def __init__(self) -> None:
        self.Sym: str = None
        self.Ene: float = None
        self.Spin: str = None
        self.Occup: float = None

        self.mo_coef_s: List[Any] = []



class molden_mol():
    """
    该类用于初始化一个molden 格式的mol文件
    格式要求见：
    https://www.theochem.ru.nl/molden/molden_format.html
    """
    def __init__(self, fp: str = None) -> None:

        # 储存单位信息
        self.unit: str = None
        # 储存坐标信息
        self.xyz: np.ndarray[np.dtype[np.float32]] = np.array([])
        # 储存原子信息
        self.atoms: List[str] = []
        # 储存Gaussian Type Orbitals信息
        self.gto_s: List[Any] = []

        # 储存molecular orbitals信息
        self.mo_s: List[Any] = []

        self.KEY_FLAG_s: Dict[str, Callable] = {"[Molden Format]":self.func_void
                                               , "[Title]": self.func_void
                                               , "[Atoms]": self.func_atoms
                                               , "[GTO]": self.func_gto
                                               , "[MO]": self.func_mo}

        self.homo_lumo_idx: List[List[int]] = None

        # 严禁修改
        self.KEY_FLAG_s = MappingProxyType(self.KEY_FLAG_s)
        #rprint(list(self.KEY_FLAG_s))

        if fp:
            self.molden_from_file(fp)

            # 这种情况肯定是读取的有内容
            self.homo_lumo_idx = self.__init_homo_lumo__()

        # 生成索引，这个索引是不同原子对应的原子轨道的idx
        self.gto_atm_idx = self.__init_gto_idx__()

    def molden_from_file(self, fp: str) -> None:

        assert os.path.exists(fp), "Fatal Error {} is not exist.".format(fp)

        with open(fp, "r+") as F:
            lines: List[str] = [ss.rstrip("\n") for ss in F.readlines()]

        cur_idx: int = -1
        old_idx: int = 0

        n_lines = len(lines)
        for cur_idx, t in enumerate(lines):


            if old_idx == n_lines - 1:
                # 已经遍历结束
                break

            elif cur_idx < old_idx:
                # 这些已经能够读取过了
                continue

            if t.startswith("["):
                _find_flag: bool = False

                for _k in list(self.KEY_FLAG_s.keys()):
                    if _k in t:
                        # 需要调用对应的函数
                        _func: Callable = self.KEY_FLAG_s[_k]
                        _flag = _k
                        _find_flag = True
                        break
                if not _find_flag:
                    raise ValueError("Fatal Error: \"{}\" is not in KEY_FLAG_s".format(t))

                old_idx = _func(flag = _flag, lines = lines, old_idx = cur_idx, end_idx = n_lines - 1)


    def func_void(self, flag: str, lines: List[str], old_idx: int, end_idx: int) -> int:
        return old_idx

    def func_atoms(self, flag: str, lines: List[str], old_idx: int, end_idx: int) -> int:

        for i, t in enumerate(lines[old_idx:], start = old_idx):
            if t.startswith("[") and flag in t:
                self.unit = t.split(" ")[1]
            elif t.startswith("[") and flag not in t:
                # 遇到下一个标识符, 停止
                return i
            elif i == end_idx:
                return i

            else:
                #print("func_atoms")
                # 这里读取的是原子, 序号, 原子序号, x, y, z
                _var: List[str] = [_s for _s in t.split(" ") if _s != ""]
                self.atoms.append(_var[0])
                _xyz = np.array([eval(_s) for _s in  _var[-3:]])
                if xyz_num := _xyz.shape[0] != 3:
                    raise ValueError("Fatal Error: xyz_num != 3")
                else:
                    self.xyz = np.concatenate([self.xyz, _xyz], axis = 0)

    def func_gto(self, flag: str, lines: List[str], old_idx: int, end_idx: int) -> int:

        def is_int_num(s: str) -> bool:
            try:
                int(s)
                return True
            except:
                return False

        cur_gto = GTO()
        for i, t in enumerate(lines[old_idx:], start = old_idx):
            if t.startswith("[") and flag in t:
                # 跳过本行
                continue
            elif (t.startswith("[") and flag not in t) or (i == end_idx):
                # 遇到下一个标识符, 或者到达末尾
                if cur_gto.atom_sequence_number != None:
                    self.gto_s.append(cur_gto)
                return i
            elif (iaw_t := t.replace(" ", "")) in ["", " ", "\t", "\n"]:
                # 储存
                self.gto_s.append(cur_gto)
                # 此时需要创建一个新的GTO用于储存
                cur_gto = GTO()
                continue
            else:
                #print("iaw_t",iaw_t)
                _var = [_s for _s in t.split(" ") if _s != ""]

                if (_n_var := len(_var) == 2) and is_int_num(_var[0]) and is_int_num(_var[1]):
                    #print("gto_s number", len(self.gto_s))
                    cur_gto.atom_sequence_number = int(_var[0])
                    cur_gto.angular_quantum_number = int(_var[1])

                elif (_n_var := len(_var) == 3):
                    # 取出目前的参数进行更新
                    # 这个时候应该初始化一个新的basis_args类

                    cur_basis_args = basis_args()
                    cur_basis_args.shell_label = _var[0]
                    cur_basis_args.number_of_primitives = int(_var[1])
                    cur_basis_args.scaling_factor_of_basis_function = float(_var[2])
                    cur_gto.basis_args.append(cur_basis_args)

                else:
                    # 这里读取的就是gaussian的两个参数
                    #rprint("string: [{}]".format(t), _var, )
                    cur_basis_args = cur_gto.basis_args[-1]
                    cur_basis_args.basis_gaussian_args_s.append(basis_gaussian_args())
                    # 取出最新的一个
                    cur_basis_gaussian_args = cur_basis_args.basis_gaussian_args_s[-1]
                    cur_basis_gaussian_args.exponential_factor = float(_var[0])
                    cur_basis_gaussian_args.contraction_coefficient = float(_var[1])


    def func_mo(self, flag: str, lines: List[str], old_idx: int, end_idx: int) -> int:

        cur_mo = MO()
        for i, t in enumerate(lines[old_idx:], start = old_idx):
           if t.startswith("[") and flag in t:
               #print(1, t)
               # 跳过本行
               continue
           elif (t.startswith("[") and flag not in t) or (i == end_idx):
               #print(2, t)
               # 遇到下一个标识符, 或者到达末尾
               # 这里存在一种情况, 此时, 已经到达文件的末尾, 但是最后一行是有数据的
               if t != "":
                   _var = [_s for _s in t.split(" ") if _s != ""]
                   cur_mo_coef = cur_mo.mo_coef_s[-1]
                   cur_mo_coef.ao_number_index = int(_var[0])
                   cur_mo_coef.mo_coefficient = float(_var[1])

                   self.mo_s.append(cur_mo)

               return i

           elif t.startswith("Sym="):
               #print(3, t, cur_mo.Ene, cur_mo.Ene != None, i, old_idx)
               _var = [_s for _s in t.split(" ") if _s != ""]
               if i == old_idx + 1:
                   cur_mo.Sym = _var[1]
                   #print(3.1,  cur_mo.Sym, len(self.mo_s))
               else:
                   self.mo_s.append(cur_mo)
                   # 此时需要创建一个新的MO用于储存
                   cur_mo = MO()
                   cur_mo.Sym = _var[1]
                   #print(3.2,  cur_mo.Sym, self.mo_s[-1].Sym)

               # 无论哪种情况都可以跳出下面的代码了
               continue

           elif t.startswith("Ene="):
               #print("5",t)
               _var = [_s for _s in t.split(" ") if _s != ""]
               cur_mo.Ene = float(_var[1])
           elif t.startswith("Spin="):
               #print("6",t)
               _var = [_s for _s in t.split(" ") if _s != ""]
               cur_mo.Spin = _var[1]
           elif t.startswith("Occup="):
               #print("7",t)
               _var = [_s for _s in t.split(" ") if _s != ""]
               cur_mo.Occup = float(_var[1])
           else:
               _var = [_s for _s in t.split(" ") if _s != ""]
               # 创建一个mo_coefficient类
               cur_mo_coef = mo_coefficient()
               cur_mo_coef.ao_number_index = int(_var[0])
               cur_mo_coef.mo_coefficient = float(_var[1])
               cur_mo.mo_coef_s.append(cur_mo_coef)

    def print_MO(self, idx: int = None) -> None:
        if idx != None:
            i_mo = self.mo_s[idx]
            print("Sym:{}, Ene: {:6f}, Spin: {}, Occup: {:.4f}".format(i_mo.Sym, i_mo.Ene, i_mo.Spin, i_mo.Occup))
            for j_mo_coef in i_mo.mo_coef_s:
                print("AO_idx: {}, MO_coef: {:.4f}".format(j_mo_coef.ao_number_index, j_mo_coef.mo_coefficient))
        else:
            for i_mo in self.mo_s:
                print("Sym:{}, Ene: {:6f}, Spin: {}, Occup: {:.4f}".format(i_mo.Sym, i_mo.Ene, i_mo.Spin, i_mo.Occup))
                for j_mo_coef in i_mo.mo_coef_s:
                    print("AO_idx: {}, MO_coef: {:.4f}".format(j_mo_coef.ao_number_index, j_mo_coef.mo_coefficient))
                    
    def print_GTO(self, idx: int = None) -> None:
        if idx != None:
            i_gto = self.gto_s[idx]
            print("Atom_idx: {}, Ang_qn: {}:".format(i_gto.atom_sequence_number, i_gto.angular_quantum_number))
            for j_basis_args in i_gto.basis_args:
                print("\tshell: {}, np: {}, sfbf: {}".format(j_basis_args.shell_label, j_basis_args.number_of_primitives, j_basis_args.scaling_factor_of_basis_function))
                for h_basis_gaussian_args in j_basis_args.basis_gaussian_args_s:
                    print("\t\texponential_factor: {:.4f}, contraction_coefficient: {:.4f}".format(h_basis_gaussian_args.exponential_factor, h_basis_gaussian_args.contraction_coefficient))

        else:
            for i_gto in self.gto_s:
                print("Atom_idx: {}, Ang_qn: {}:".format(i_gto.atom_sequence_number, i_gto.angular_quantum_number))
                for j_basis_args in i_gto.basis_args:
                    print("\tshell: {}, np: {}, sfbf: {}".format(j_basis_args.shell_label, j_basis_args.number_of_primitives, j_basis_args.scaling_factor_of_basis_function))
                    for h_basis_gaussian_args in j_basis_args.basis_gaussian_args_s:
                        print("\t\texponential_factor: {:.4f}, contraction_coefficient: {:.4f}".format(h_basis_gaussian_args.exponential_factor, h_basis_gaussian_args.contraction_coefficient))

    def print_info(self, flag: str):
        if flag == "MO":
            pass
    
    def FO(self) -> None:
        """
        这个函数内部做过反转，所以，HOMO_0就是最高占据，LUMO_0就是最低空轨
        """
        
        assert self.homo_lumo_idx != None, "Error homo_lumo_idx is None"
        
        _homo, _lumo = self.homo_lumo_idx
        lumo_ene = []
        for i, idx in enumerate(_lumo):
            i_mo = self.mo_s[idx]
            lumo_ene.append(i_mo.Ene * 27.2114)         # eV
        homo_ene = []
        for i, idx in enumerate(_homo[::-1]):
            i_mo = self.mo_s[idx] 
            homo_ene.append(i_mo.Ene * 27.2114)         # eV
        
        lumo_ene = np.array(lumo_ene, dtype = np.float64)
        homo_ene = np.array(homo_ene, dtype = np.float64)
        
            
        return lumo_ene, homo_ene
    
    def print_FO(self) -> None:
        """
        该函数用于输出前线轨道的内容
        """
        Alpha_sign = "[yellow]-[/yellow][green]↿[/green][yellow]-[/yellow] [yellow]- -[/yellow]"
        Beta_sign = "[yellow]- -[/yellow] [yellow]-[/yellow][green]⇂[/green][yellow]-[/yellow]"
        Full_sign = "[yellow]-[/yellow][green]↿[/green][yellow]-[/yellow] [yellow]-[/yellow][green]⇂[/green][yellow]-[/yellow]"
        None_sign = "[yellow]- -[/yellow] [yellow]- -[/yellow]"

        assert self.homo_lumo_idx != None, "Error homo_lumo_idx is None"

        # 倒序打印
        _homo, _lumo = self.homo_lumo_idx
        # 创建表格
        table = Table(title="Orbital Energies and Occupations", title_style="bold magenta")
        table.add_column("#", justify="center", style="cyan")
        table.add_column("Occupation", justify="center", style="yellow")
        table.add_column("Energy/Eh", justify="center", style="green")
        table.add_column("Energy/eV", justify="center", style="white")
        table.add_column("Label", justify="center", style="red")

        # 添加分隔线
        table.add_row("-" * 10, "-" * 20, "-" * 20, "-" * 20, "-" * 10)

        n_lumo = len(_lumo) - 1
        for i, idx in enumerate(_lumo[::-1]):
            i_mo = self.mo_s[idx]
            label = Text(f"LUMO_{n_lumo - i}", style="bold magenta")

            if i_mo.Occup > 1:
                occupation = Full_sign
            elif i_mo.Occup == 1:
                occupation = Alpha_sign if i_mo.Spin == "Alpha" else Beta_sign
            else:
                occupation = None_sign

            table.add_row(
                str(i + 1),
                occupation,
                f"{i_mo.Ene:.6f}",
                f"{i_mo.Ene * 27.2114:.4f}",  # Convert Eh to eV
                label
            )

        for i, idx in enumerate(_homo[::-1]):
            i_mo = self.mo_s[idx]
            label = Text(f"HOMO_{i}", style="bold magenta")

            if i_mo.Occup > 1:
                occupation = Full_sign
            elif i_mo.Occup == 1:
                occupation = Alpha_sign if i_mo.Spin == "Alpha" else Beta_sign
            else:
                occupation = None_sign

            table.add_row(
                str(i + 1),
                occupation,
                f"{i_mo.Ene:.6f}",
                f"{i_mo.Ene * 27.2114:.4f}",  # Convert Eh to eV
                label
            )

        # 添加分隔线
        table.add_row("-" * 10, "-" * 20, "-" * 20, "-" * 20, "-" * 10)
        rprint(table)

    def __init_homo_lumo__(self) -> List[List[int]]:
        """
        开壳层体系需要区分Alpha与Beta
        闭壳层不需要
        该函数返回一个2D列表
            前半部分储存homo的索引, 后半部分储存lumo的索引
        """


        var_list: List[Dict[str, Any]] = []
        for i, i_mo in enumerate(self.mo_s):
            var_list.append(
                    {
                        "idx": i
                       ,"Ene": i_mo.Ene
                       ,"Occup":i_mo.Occup
                       ,"Spin":i_mo.Spin
                        }
                    )
        # sorted
        var_list = sorted(var_list, key=lambda x: x['Ene'])

        # 划分HOMO与LOMU:

        _homo: List[int] = None
        _lumo: List[int] = None

        split_index: int = None
        for i in range(len(var_list) - 1):
            if var_list[i+1]['Occup'] < 1:
                split_index = i + 1
                break

        # 之返回idx就可以了
        _homo = [x["idx"] for x in var_list[:split_index]]
        _lumo = [x["idx"] for x in var_list[split_index:]]

        return  [_homo, _lumo]

    def __init_gto_idx__(self) -> List[int]:
        """
        需要根据self.gto_s中的每个原子信息,生成一个三维的列表用于储存索引
        例如:
            [[[0], [1, 2, 3]], [[4], [5, 6, 7]], ...]
            s, p, s, p, ...
        """
        count = 0
        out = []
        for idx in range(len(self.gto_s)):
            i_gto = self.gto_s[idx]
            i_out = []
            for j_basis_args in i_gto.basis_args:
            
                if j_basis_args.shell_label == "s":
                    i_out.append(count)
                    count += 1
                elif j_basis_args.shell_label == "p":
                    _out = []
                    for _ in range(3):
                        _out.append(count)
                        count+=1
                    i_out.append(_out)
                elif j_basis_args.shell_label == "d":
                    _out = []
                    for _ in range(5):
                        _out.append(count)
                        count+=1
                    i_out.append(_out)
                
                
            out.append(i_out)
        return out
                
             
                    
    def mol_homo(self, idx: int = None):
        pass

    def mo_atm_contribution(self, mo_idx: tuple[bool, int], atm_idx: int) -> np.float64:
        """
        bool1: False LUMO
        bool2: True  HOMO
        """
        assert self.homo_lumo_idx != None, "Error homo_lumo_idx is None"
        _homo, _lumo = self.homo_lumo_idx
        
        flag, idx = mo_idx
        if flag:
            idx_2 = _homo[::-1][idx]
        else:
            idx_2 = _lumo[idx]
        
        i_mo = self.mo_s[idx_2]
        mo_coefficient_arr = []
        #print("Sym:{}, Ene: {:6f}, Spin: {}, Occup: {:.4f}".format(i_mo.Sym, i_mo.Ene, i_mo.Spin, i_mo.Occup))
        for j_mo_coef in i_mo.mo_coef_s:
            #print("AO_idx: {}, MO_coef: {:.4f}".format(j_mo_coef.ao_number_index, j_mo_coef.mo_coefficient))
            mo_coefficient_arr.append(j_mo_coef.mo_coefficient)
        
        mo_coefficient_arr = np.array(mo_coefficient_arr)
        
        mo_coefficient_arr_2 = mo_coefficient_arr**2
        mo_coefficient_arr_2_sum = np.sum(mo_coefficient_arr_2)
        mo_coefficient_arr_norm = mo_coefficient_arr_2 / mo_coefficient_arr_2_sum
        
        atm_orb_idx = self.gto_atm_idx[atm_idx]
        # 合并不同的shell为一个list
        
        def flatten_list(list_: List):
            flat_list = []
            for item in list_:
                if isinstance(item, list):
                    flat_list.extend(flatten_list(item))
                else:
                    flat_list.append(item)
            return flat_list   
             
        flat_atm_orb_idx = flatten_list(atm_orb_idx)
        
        return np.sum(mo_coefficient_arr_norm[flat_atm_orb_idx])
        
        
if __name__ == "__main__":
    fp: str = "/data/node7/iaw/DATA/inverted-SF/dataset/ppp-invest/data_xtb_cal/IAW000082809/molden.input"
    test = molden_mol(fp = fp)

    print("unit:")
    rprint(test.unit)
    print("atoms:")
    rprint(test.atoms)
    print("xyz:")
    rprint(test.xyz)

    print("Atom Orbital idx based on shell (s, p, d)")
    rprint(test.gto_atm_idx)
    
    #test.print_MO(idx = 0)
    #test.print_MO(idx = 1)
    #test.print_MO(idx = 2)
    #test.print_MO()

    #test.print_GTO(idx = 0)
    #test.print_GTO(idx = 1)

    test.print_FO()
    
    # FO function
    lumo_ene, homo_ene = test.FO()
    print("LUMO")
    rprint(lumo_ene)
    print("HOMO")
    rprint(homo_ene)

    #print("homo_lumo_idx")
    #rprint(test.homo_lumo_idx)
    
    #print("atom -> HOMO")
    #for i, i_atm in enumerate(test.atoms):
    #   rprint("atom[{}]({})".format(i_atm, i), "{:.4f}".format(test.mo_atm_contribution((True, 0), i)))




