from dataclasses import dataclass
from rich import print as rp
import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray
from rdkit.Chem import GetPeriodicTable
from copy import deepcopy
from scipy.interpolate import RegularGridInterpolator

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from util.tool import timeit
import math

#++++++++++++++++++++++++++++++++++++++++++# 解析xtb长生的cube数据 #+++++++++++++++++++++++++++++++++++++++++++++++++++#
# CUBE: 数据格式
# Cube_Reader: 从cube_fp中读取并实例化一个CUBE
# gen_grid_xyz: 根据CUBE中的数据, 产生网格的坐标
# smooth_grid_cube_scale: 根据scale, 对CUBE中的数据进行平滑, 返回一个新的CUBE
# smooth_grid_cube_box: 根据box的信息, 对CUBE中的数据进行平滑, 返回一个新的CUBE
# grid_xyz_to_box: 根据给定的box, 将分子坐标以及网格变换过去, 注意, 这里的单位是bohr
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#



@dataclass
class CUBE:
    n_atm: int              # 原子数目
    origin: NDArray         # 原点坐标
    n_grid: List            # Nx, Ny, Nz每个方向上点的数目
    lattice_vec: NDArray    # xyz轴的向量
    atoms: List             # 储存原子名称, 全大写
    coor: NDArray           # 储存原子坐标
    grid_data: NDArray      # 储存格点上的数值


def Cube_Reader(fp: str) -> CUBE:
    """
    要求是xtb的cub文件
    """
    def __line_read__(ss: str) -> Tuple[int, List]:
        out = []
        ss = ss.rstrip("\n").replace("\t", " ").replace("  ", " ").split(" ")
        ss = [s for s in ss if s != ""]
        #rp(ss)
        for s in ss:
            out.append(eval(s))

        return len(out), out

    cube = CUBE(n_atm = 0,
                origin = None,
                n_grid = None,
                lattice_vec = None,
                atoms = None,
                coor = None,
                grid_data = None)
    
    with open(fp, "r") as F:
        lines = F.readlines()
        # 跳过两行的title
        line_1 = lines[2]
        line_2 = lines[3:6]
        lines_3 = lines[6:]


        # 解析原子数目与格点数据的坐标原点
        line_out = __line_read__(line_1)[-1]
        cube.n_atm = line_out[0]
        
        #assert cube.n_atm > 0, rp("Error\\[iaw]>: please check the number of atoms in cub file.")
        cube.origin = np.array(line_out[1:], dtype=np.float64)

        if cube.n_atm >= 0:
            lines_4 = lines[6+cube.n_atm:]
        else:
            cube.n_atm = abs(cube.n_atm)
            lines_4 = lines[6+cube.n_atm+1:]            # 临时补丁, 支持单个分子轨道
            
        # 解析网格参数
        line_out =[__line_read__(_l)[-1] for _l in line_2]
        cube.n_grid = [i[0] for i in line_out]
        lattice_vec = [i[1:] for i in line_out]
        for i in lattice_vec:
            assert len(i) ==3, rp("Error\\[iaw]>: please check the grid in cub file.") 
        cube.lattice_vec = np.array(lattice_vec, dtype=np.float64)

        # 解析原子类型以及原子坐标
        periodic_table = GetPeriodicTable()
        atoms = []
        coor = []
        for i in range(cube.n_atm):
            n_l, _l_out = __line_read__(lines_3[i])
            assert n_l == 5, rp("Error\\[iaw]>:please check the atom sym and xyz in cub file")
            i_sym = periodic_table.GetElementSymbol(_l_out[0])      #.upper()
            atoms.append(i_sym)
            i_xyz = _l_out[2:]
            coor.append(i_xyz)
        cube.atoms = atoms
        cube.coor = np.array(coor)

        # 解析格点
        n_tot_point = cube.n_grid[0] * cube.n_grid[1] * cube.n_grid[2]
        grid_data = []
        for _l in lines_4:
            grid_data.extend(__line_read__(_l)[-1])
        #rp(len(grid_data))
        grid_data = np.array(grid_data, dtype=np.float64).reshape(*cube.n_grid)
        cube.grid_data = grid_data

    return cube

@timeit         
def gen_grid_xyz(cub: CUBE) -> Tuple[NDArray, NDArray, NDArray]:
    nx, ny, nz = cub.n_grid
    # 广播成 3D 网格, 保证 i 对应 x，j 对应 y，k 对应 zx
    I, J, K = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float), np.arange(nz, dtype=float), indexing='ij')

    X = cub.origin[0] + I * cub.lattice_vec[0, 0] + J * cub.lattice_vec[1, 0] + K * cub.lattice_vec[2, 0]
    Y = cub.origin[1] + I * cub.lattice_vec[0, 1] + J * cub.lattice_vec[1, 1] + K * cub.lattice_vec[2, 1]
    Z = cub.origin[2] + I * cub.lattice_vec[0, 2] + J * cub.lattice_vec[1, 2] + K * cub.lattice_vec[2, 2]

    return X, Y, Z

@timeit 
def smooth_grid_cube_scale(cub: CUBE, scale: float, method: str = "cubic") -> CUBE:

    """
    这里需要转化为坐标(bohr), 如果用索引直接增加点, 误差很大 
    """
    smooth_cub = deepcopy(cub)
    nx, ny, nz = cub.n_grid
    
    # 修正：用物理坐标替代索引作为插值基准
    X_org, Y_org, Z_org = gen_grid_xyz(cub)
    x_axis = X_org[:, 0, 0].astype(np.float64)
    y_axis = Y_org[0, :, 0].astype(np.float64)
    z_axis = Z_org[0, 0, :].astype(np.float64)
    original_grid = (x_axis, y_axis, z_axis)

    # 新网格的物理坐标范围：和原始网格一致，仅加密
    x_min, x_max = x_axis.min(), x_axis.max()
    y_min, y_max = y_axis.min(), y_axis.max()
    z_min, z_max = z_axis.min(), z_axis.max()
    
    new_nx = int(nx * scale)
    new_ny = int(ny * scale)
    new_nz = int(nz * scale)
    
    # 修正：生成新的物理坐标网格（而非索引）
    new_grid = (
        np.linspace(x_min, x_max, new_nx, dtype=np.float64),
        np.linspace(y_min, y_max, new_ny, dtype=np.float64),
        np.linspace(z_min, z_max, new_nz, dtype=np.float64)
    )

    interpolator = RegularGridInterpolator(
        points=original_grid,
        values=cub.grid_data,
        method=method,
        bounds_error=False, 
        fill_value=0.0
    )

    xi, yi, zi = np.meshgrid(*new_grid, indexing='ij')
    new_points = np.stack([xi.flatten(), yi.flatten(), zi.flatten()], axis=1)
    smooth_grid_data = interpolator(new_points).reshape(new_nx, new_ny, new_nz)
    
    # 修正：晶格向量基于物理区间计算（而非除法）
    smooth_cub.n_grid = [new_nx, new_ny, new_nz]
    smooth_cub.lattice_vec = np.array([
        [(x_max - x_min)/(new_nx-1) if new_nx>1 else 0, 0, 0],
        [0, (y_max - y_min)/(new_ny-1) if new_ny>1 else 0, 0],
        [0, 0, (z_max - z_min)/(new_nz-1) if new_nz>1 else 0]
    ], dtype=np.float64)
    smooth_cub.grid_data = smooth_grid_data

    return smooth_cub


def smooth_grid_cube_box(cub: CUBE, box: Tuple[float, float, float, float, float, float], n_grid: Tuple[int, int, int], method: str = "cubic") -> CUBE:

    smooth_cub = deepcopy(cub)
    X_org, Y_org, Z_org = gen_grid_xyz(cub)
    x_axis = X_org[:, 0, 0].astype(np.float64)
    y_axis = Y_org[0, :, 0].astype(np.float64)
    z_axis = Z_org[0, 0, :].astype(np.float64)
    original_grid = (x_axis, y_axis, z_axis)

    x_min, y_min, z_min, x_max, y_max, z_max = box
    x_len = x_max - x_min
    y_len = y_max - y_min
    z_len = z_max - z_min
    new_origin = np.array([x_min, y_min, z_min], dtype=np.float64)

    nx_new, ny_new, nz_new = n_grid
    x_new = np.linspace(x_min, x_max, nx_new, dtype=np.float64)
    y_new = np.linspace(y_min, y_max, ny_new, dtype=np.float64)
    z_new = np.linspace(z_min, z_max, nz_new, dtype=np.float64)
    new_grid = (x_new, y_new, z_new)

    interpolator = RegularGridInterpolator(
        points=original_grid,
        values=cub.grid_data,
        method=method,  
        bounds_error=False, 
        fill_value=0.0  
    )
    xi, yi, zi = np.meshgrid(*new_grid, indexing='ij')
    new_points = np.stack([xi.flatten(), yi.flatten(), zi.flatten()], axis=1)
    new_grid_data = interpolator(new_points).reshape(nx_new, ny_new, nz_new)
    smooth_cub.n_grid = n_grid
    smooth_cub.origin = new_origin
    smooth_cub.lattice_vec = np.array([
            [x_len / (nx_new - 1) if nx_new > 1 else 0, 0, 0], 
            [0, y_len / (ny_new - 1) if ny_new > 1 else 0, 0], 
            [0, 0, z_len / (nz_new - 1) if nz_new > 1 else 0]  
            ], dtype=np.float64)
    smooth_cub.grid_data = new_grid_data
    return smooth_cub
