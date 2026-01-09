from typing import List, Tuple
import sys
import numpy as np
from numpy.typing import NDArray
from copy import deepcopy
import os
from skimage.measure import marching_cubes

from vispy import app, scene, color
from vispy.geometry.isosurface import isosurface
from vispy.scene.visuals import Mesh, Markers, Line, Sphere
from vispy.geometry import create_cylinder
from PyQt6.QtWidgets import (
    QApplication, QVBoxLayout, QWidget, QLabel, QLineEdit,
    QHBoxLayout, QPushButton, QGroupBox, QGridLayout
)
from PyQt6.QtGui import QIntValidator
from PyQt6.QtCore import Qt

from pathlib import Path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
from util.cub_tools import Cube_Reader, gen_grid_xyz, smooth_grid_cube_scale
from util.mol import get_color_radii, build_bond
from util.constant import BOHR_TO_A
from util.tool import timeit
from config.theme import default_theme as apply_theme

import argparse

from rich import print as rp


def init_atoms(coor_A: NDArray, color_s: NDArray, radii_s: NDArray):

    atom_spheres = []
    
    for idx, (coor, color, radius) in enumerate(zip(coor_A, color_s, radii_s)):
        #rp("Info\\[iaw]>: the sphere color is {}".format(color))
        sphere = Sphere(
            radius=radius*apply_theme["sphere_scale"],        # 缩放半径至合适大小
            subdivisions=apply_theme["sphere_subdivisions"],  # 细分次数（值越大球体越平滑）
            method='ico',                                     # ico方法生成的球体更均匀
            color=color,                                      # RGBA颜色（完全不透明）
            edge_color=None,
            shading=apply_theme["atom_shading"]                                  # 平滑着色
        )
        
        # 设置球体位置
        sphere.set_gl_state('translucent', depth_test=True, cull_face=False, blend = True)
        sphere.transform = scene.transforms.MatrixTransform()
        sphere.transform.translate(coor)
        
        atom_spheres.append(sphere)
    
    return atom_spheres


def init_bonds(bond_s, coor_A):
    
    bonds_lines = []
    for i, j in bond_s:
        try:
            p1, p2 = coor_A[i], coor_A[j]
            p1 = np.array(p1)
            p2 = np.array(p2)

            # 计算化学键方向和长度
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue
            
            # 计算圆柱的中点和方向
            center = (p1 + p2) / 2
            axis = direction / length

            # 创建圆柱
            radius = apply_theme["bond_radius"]  # 化学键半径
            cylinder = create_cylinder(
                radius=(radius, radius),
                length=length,
                rows=10,
                cols=20
            )
            vertices = cylinder.get_vertices()
            faces = cylinder.get_faces()

            # 旋转和平移圆柱到正确位置
            if np.allclose(axis, [0, 0, 1]):
                rotation = np.eye(3)
            else:
                # 计算旋转轴和角度
                up = np.array([0, 0, 1])
                cross = np.cross(up, axis)
                dot = np.dot(up, axis)
                angle = np.arccos(dot)
                if np.linalg.norm(cross) < 1e-6:
                    rotation = np.eye(3)
                else:
                    cross /= np.linalg.norm(cross)
                    # 罗德里格斯旋转公式
                    K = np.array([
                        [0, -cross[2], cross[1]],
                        [cross[2], 0, -cross[0]],
                        [-cross[1], cross[0], 0]
                    ])
                    rotation = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

            # 应用旋转和平移
            # 注意：create_cylinder创建的圆柱体默认是沿z轴方向的，长度为length
            # 所以我们需要先将圆柱体的中心移动到原点，然后旋转，最后平移到center
            # 1. 将圆柱体的中心移动到原点
            vertices -= np.array([0, 0, length / 2])
            # 2. 旋转
            vertices = vertices @ rotation.T
            # 3. 平移到center
            vertices += center

            # 创建Mesh对象
            cyl = Mesh(
                vertices=vertices,
                faces=faces,
                color=apply_theme["bond_color"],
                shading=apply_theme["bond_shading"],
                )
            cyl.set_gl_state('translucent', depth_test=True, cull_face=False, blend = True)
            bonds_lines.append(cyl)
        except Exception as e:
            rp("Error\\[iaw]>: draw bond: {}-{}: {}".format(i, j, e))
            os.exit(1)
    return bonds_lines

@timeit
def init_cube_scatter(scale: int, smooth_method: str, cut: float = 0.0002, band: float = 1e-6):

    global ECLOUD_CUBE, ECLOUD_CACHE, ECLOUD_MESH, ECLOUD_SCATTER
    
    if "ECLOUD_SCALE_{}".format(scale) in ECLOUD_CACHE.keys():
        COORDS = ECLOUD_CACHE["ECLOUD_SCALE_{}".format(scale)]["COORDS"]
        ECLOUD_VALUE = ECLOUD_CACHE["ECLOUD_SCALE_{}".format(scale)]["ECLOUD_VALUE"]
        ECLOUD_VALUE_flat = ECLOUD_VALUE.flatten()
   
    elif scale == 1:
        ECLOUD_VALUE = ECLOUD_CUBE.grid_data
        COORDS = gen_grid_xyz(ECLOUD_CUBE)
        ECLOUD_VALUE_flat = ECLOUD_VALUE.flatten()
        ECLOUD_CACHE["ECLOUD_SCALE_{}".format(scale)] = {
            "COORDS": COORDS,
            "ECLOUD_VALUE": ECLOUD_VALUE
        }        
    else:
        cub_smoothed = smooth_grid_cube_scale(ECLOUD_CUBE, scale=scale, method=smooth_method)
        COORDS = gen_grid_xyz(cub_smoothed)
        ECLOUD_VALUE = cub_smoothed.grid_data
        ECLOUD_VALUE_flat = ECLOUD_VALUE.flatten()
        ECLOUD_CACHE["ECLOUD_SCALE_{}".format(scale)] = {
            "COORDS": COORDS,
            "ECLOUD_VALUE": ECLOUD_VALUE
        }

    mask = np.abs(ECLOUD_VALUE_flat - cut) < band
    X_flat = COORDS[0].ravel()
    Y_flat = COORDS[1].ravel()
    Z_flat = COORDS[2].ravel()
    POS_MASK = np.column_stack((X_flat[mask], Y_flat[mask], Z_flat[mask])) * BOHR_TO_A
    value_mask = ECLOUD_VALUE_flat[mask]

    rp("Info\\[iaw]>: scale: {}, len(value) = {}".format(scale, value_mask.size))
    if value_mask.size == 0:
        COLOR_RGB = np.array([])
    else:
        val_norm = (value_mask - value_mask.min()) / (value_mask.max() - value_mask.min() + 1e-8)
        cmap = color.get_colormap('viridis')
        COLOR_RGB = cmap.map(val_norm)[:, :3]
        

    if COLOR_RGB.size == 0:
        ECLOUD_SCATTER.set_data(COLOR_RGB)
        pass
    else:
        ECLOUD_SCATTER.set_data(POS_MASK, face_color=COLOR_RGB, edge_color=None, size=apply_theme["scatter_size"])
    ECLOUD_SCATTER.visible = True

    if ECLOUD_MESH:
        ECLOUD_MESH.visible = False

@timeit
def init_cube_mesh(scale: int, smooth_method: str, cut: float = 0.005, band: float = 1e-6):
 
    global ECLOUD_CUBE, ECLOUD_CACHE, ECLOUD_MESH, ECLOUD_SCATTER

    # 先查询缓存
    if "ECLOUD_SCALE_{}".format(scale) in ECLOUD_CACHE.keys():
        COORDS = ECLOUD_CACHE["ECLOUD_SCALE_1"]["COORDS"]
        ECLOUD_VALUE = ECLOUD_CACHE["ECLOUD_SCALE_1"]["ECLOUD_VALUE"]

    elif scale == 1:
        if "ECLOUD_SCALE_1" not in ECLOUD_CACHE.keys():
            rp("Error\\[iaw]>: Please run scatter plot first to cache the original data.")
        else:
            COORDS = ECLOUD_CACHE["ECLOUD_SCALE_1"]["COORDS"]
            ECLOUD_VALUE = ECLOUD_CACHE["ECLOUD_SCALE_1"]["ECLOUD_VALUE"]
    else:
        if "SCATTER_SCALE_{}".format(scale) not in ECLOUD_CACHE.keys():
            cub_smoothed = smooth_grid_cube_scale(ECLOUD_CUBE, scale=scale, method=smooth_method)
            ECLOUD_VALUE = cub_smoothed.grid_data
            COORDS = gen_grid_xyz(cub_smoothed)
            # 为scatter创建缓存
            #rp("Info\\[iaw]>: smooth scale is {}".format(scale))
            ECLOUD_CACHE["ECLOUD_SCALE_{}".format(scale)] = {
                 "COORDS": COORDS
                , "ECLOUD_VALUE": ECLOUD_VALUE
            }
        else:
            COORDS = ECLOUD_CACHE["ECLOUD_SCALE_{}".format(scale)]["COORDS"]
            ECLOUD_VALUE = ECLOUD_CACHE["ECLOUD_SCALE_{}".format(scale)]["ECLOUD_VALUE"]

    X_s, Y_s, Z_s = COORDS
    dx = (X_s.max() - X_s.min()) / (X_s.shape[0] - 1) if X_s.shape[0] > 1 else 1.0
    dy = (Y_s.max() - Y_s.min()) / (Y_s.shape[1] - 1) if Y_s.shape[1] > 1 else 1.0
    dz = (Z_s.max() - Z_s.min()) / (Z_s.shape[2] - 1) if Z_s.shape[2] > 1 else 1.0
    origin = [X_s.min(), Y_s.min(), Z_s.min()]
    verts, FACES, normals, values = marching_cubes(
        volume=ECLOUD_VALUE,
        level=cut, 
        method='lewiner')
    verts_physical = np.copy(verts)
    verts_physical[:, 0] = origin[0] + verts[:, 0] * dx
    verts_physical[:, 1] = origin[1] + verts[:, 1] * dy
    verts_physical[:, 2] = origin[2] + verts[:, 2] * dz
    VERTS_A = verts_physical * BOHR_TO_A

    ECLOUD_MESH.set_data(
        vertices=VERTS_A,
        faces=FACES,
    )

    ECLOUD_MESH.visible = True
    if ECLOUD_SCATTER:
        ECLOUD_SCATTER.visible = False

def update_display():
    global ECLOUD_SCATTER, ECLOUD_MESH, ECLOUD_DISPLAY_MODE, ECLOUD_CUT, ECLOUD_BAND, ECLOUD_SMOOTH_FACTOR, ECLOUD_SMOOTH_METHOD
    
    try:
        if ECLOUD_SCATTER:
            ECLOUD_SCATTER.visible = False
        if ECLOUD_MESH:
            ECLOUD_MESH.visible = False
        
        if ECLOUD_DISPLAY_MODE == "scatter":
            init_cube_scatter(scale= ECLOUD_SMOOTH_FACTOR, smooth_method = ECLOUD_SMOOTH_METHOD, cut=ECLOUD_CUT, band=ECLOUD_BAND)
        elif ECLOUD_DISPLAY_MODE == "mesh": 
            init_cube_mesh(scale= ECLOUD_SMOOTH_FACTOR, smooth_method = ECLOUD_SMOOTH_METHOD, cut=ECLOUD_CUT, band=ECLOUD_BAND)
    except Exception as e:
        rp("Error\\[iaw]>: Update display Error: {}".format(e))       
        

class ControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        global ECLOUD_CUT, ECLOUD_BAND, ECLOUD_SMOOTH_FACTOR
        self.setWindowTitle("SetUp")
        self.setMinimumWidth(600)  # 设置最小宽度，避免控件挤压

        main_layout = QVBoxLayout()
        # 全局布局间距最小化
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # 1. 显示模式 + 重置视角（第一行）
        mode_group = QGroupBox("显示模式")
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(5)
        
        self.scatter_btn = QPushButton("散点", minimumWidth=60)
        self.mesh_btn = QPushButton("3D等值面", minimumWidth=80)
        self.reset_view_btn = QPushButton("重置视角", minimumWidth=70)
        
        self.scatter_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.scatter_btn.clicked.connect(lambda: self.change_display_mode("scatter"))
        self.mesh_btn.clicked.connect(lambda: self.change_display_mode("mesh"))
        self.reset_view_btn.clicked.connect(self.reset_view)
        
        mode_layout.addWidget(self.scatter_btn)
        mode_layout.addWidget(self.mesh_btn)
        mode_layout.addWidget(self.reset_view_btn)
        mode_layout.addStretch()  
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)

        # 2. 等值面+带宽+平滑设置+Run按钮（合并为一行）
        param_group = QGroupBox("SETUP")
        param_layout = QHBoxLayout()
        param_layout.setSpacing(5)
        
        # 2.1 等值面阈值 Cut
        param_layout.addWidget(QLabel("IsoValue:"), alignment=Qt.AlignmentFlag.AlignRight)
        self.cut_input = QLineEdit()
        self.cut_input.setText(f"{ECLOUD_CUT:.6f}")
        self.cut_input.setFixedWidth(70)
        param_layout.addWidget(self.cut_input)
        
        # 2.2 带宽 Band
        param_layout.addWidget(QLabel("Band:"), alignment=Qt.AlignmentFlag.AlignRight)
        self.band_input = QLineEdit()
        self.band_input.setText(f"{ECLOUD_BAND:.6f}")
        self.band_input.setFixedWidth(70)
        param_layout.addWidget(self.band_input)
        
        # 2.3 平滑因子
        param_layout.addWidget(QLabel("平滑因子:"), alignment=Qt.AlignmentFlag.AlignRight)
        self.smooth_input = QLineEdit()
        self.smooth_input.setText(f"{int(ECLOUD_SMOOTH_FACTOR)}")
        self.smooth_input.setFixedWidth(50)
        self.smooth_input.setValidator(QIntValidator())
        param_layout.addWidget(self.smooth_input)
        
        # 2.4 插值方法
        param_layout.addWidget(QLabel("插值:"), alignment=Qt.AlignmentFlag.AlignRight)
        self.linear_btn = QPushButton("线性（推荐）", minimumWidth=70)
        self.cubic_btn = QPushButton("立方", minimumWidth=50)
        self.linear_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.linear_btn.clicked.connect(lambda: self.change_interp_method("linear"))
        self.cubic_btn.clicked.connect(lambda: self.change_interp_method("cubic"))
        param_layout.addWidget(self.linear_btn)
        param_layout.addWidget(self.cubic_btn)
        
        # 2.5 Run按钮（核心新增：点击才更新）
        self.run_btn = QPushButton("Run", minimumWidth=60)
        self.run_btn.setStyleSheet("background-color: #2196F3; color: white;")
        self.run_btn.clicked.connect(self.on_run_click)
        param_layout.addWidget(self.run_btn)
        
        # 填充空白，控件靠左对齐
        param_layout.addStretch()
        param_group.setLayout(param_layout)
        main_layout.addWidget(param_group)

        # 保存：
        #self.screenshot_btn = QPushButton("save", minimumWidth=100)
        #self.screenshot_btn.setStyleSheet("background-color: #9C27B0; color: white;")
        #self.screenshot_btn.clicked.connect(self.save_screenshot)
        #mode_layout.addWidget(self.screenshot_btn)

        self.setLayout(main_layout)
    #def save_screenshot(self):
    #    global APP_CANVAS, APP_VIEW
    #    import imageio
#
    #    try:
    #        img = APP_CANVAS.render(alpha=True)
    #        if img.dtype != np.uint8:
    #            img = (img * 255).astype(np.uint8)
    #        save_path = "molecule_capture.png"
    #        imageio.imsave(save_path, img)
    #        rp(f"Success\\[iaw]>: Screenshot saved to {save_path}")
#
    #    except Exception as e:
    #        rp(f"Error\\[iaw]>: Screenshot failed: {e}")

    def change_display_mode(self, mode):

        global ECLOUD_DISPLAY_MODE
        ECLOUD_DISPLAY_MODE = mode
        
        self.scatter_btn.setStyleSheet("" if mode != "scatter" else "background-color: #4CAF50; color: white;")
        self.mesh_btn.setStyleSheet("" if mode != "mesh" else "background-color: #4CAF50; color: white;")
        
        update_display() 

    def change_interp_method(self, method):

        global ECLOUD_SMOOTH_METHOD
        smooth_method = method
        
        self.linear_btn.setStyleSheet("" if method != "linear" else "background-color: #4CAF50; color: white;")
        self.cubic_btn.setStyleSheet("" if method != "cubic" else "background-color: #4CAF50; color: white;")

    def reset_view(self):
        global APP_VIEW
        if APP_VIEW and APP_VIEW.camera:
            APP_VIEW.camera.reset()

    def on_run_click(self):
        global ECLOUD_CUT, ECLOUD_BAND, ECLOUD_SMOOTH_FACTOR
        try:
   
            ECLOUD_CUT = float(self.cut_input.text())
            ECLOUD_BAND = float(self.band_input.text())
            ECLOUD_SMOOTH_FACTOR = max(1, min(5, int(self.smooth_input.text())))
            update_display()
            
        except ValueError as e:

            rp("Error\\[iaw]> input num can not be supported :{}".format(e))
            self.cut_input.setText(f"{ECLOUD_CUT:.6f}")
            self.band_input.setText(f"{ECLOUD_BAND:.6f}")
            self.smooth_input.setText(f"{ECLOUD_SMOOTH_FACTOR}")

    def cut_input_change(self):
        global ECLOUD_CUT
        try:
            value = float(self.cut_input.text())
        except ValueError:
            self.cut_input.setText(f"{ECLOUD_CUT:.6f}")

    def band_input_change(self):
        global ECLOUD_BAND
        try:
            value = float(self.band_input.text())
        except ValueError:
            self.band_input.setText(f"{ECLOUD_BAND:.5f}")

    def smooth_input_change(self):
        global ECLOUD_SMOOTH_FACTOR
        try:
            value = int(self.smooth_input.text())
            value = max(1, min(5, value))
            self.smooth_input.setText(f"{value}")
        except ValueError:
            self.smooth_input.setText(f"{ECLOUD_SMOOTH_FACTOR}")


def Parm():
    parser = argparse.ArgumentParser(description='[IAWNIX]>: A visualization software for cube file information obtained from xtb.')
    parser.add_argument('-cube'
                        , type = str
                        , nargs = 1
                        , help = 'XX.cube')
    return parser.parse_args()

ECLOUD_CUBE = None
ECLOUD_CUBE_ORIGINAL = None
ECLOUD_SCATTER = None
ECLOUD_MESH = None
ECLOUD_BALLS = None
ECLOUD_BONDS = None

APP_VIEW = None
APP_CANVAS = None
ECLOUD_CACHE = {}
ECLOUD_CUT = 0.0002
ECLOUD_BAND = 1e-6
ECLOUD_SMOOTH_FACTOR = 1              # 初始无平滑
ECLOUD_SMOOTH_METHOD = "linear"       # 插值方法
ECLOUD_DISPLAY_MODE = "scatter"       # 显示模式：scatter/mesh


def main():
    parm = Parm()

    import time
    start_time = time.perf_counter()

    global ECLOUD_CUBE, ECLOUD_CUBE_ORIGINAL
    global ECLOUD_SCATTER, ECLOUD_MESH, ECLOUD_BALLS, ECLOUD_BONDS

    global APP_VIEW, APP_CANVAS
    global ECLOUD_CACHE, ECLOUD_CUT, ECLOUD_BAND, ECLOUD_SMOOTH_FACTOR, ECLOUD_SMOOTH_METHOD, ECLOUD_DISPLAY_MODE

    ECLOUD_CUBE = Cube_Reader(parm.cube[0])
    ECLOUD_CUBE_ORIGINAL = deepcopy(ECLOUD_CUBE)
    end_time1 = time.perf_counter()
    rp("Info\\[iaw]>: Load cube file time: {:.4f} seconds.".format(end_time1 - start_time))

    colors, radii = get_color_radii(ECLOUD_CUBE.atoms)
    bonds = build_bond(ECLOUD_CUBE.coor * BOHR_TO_A, ECLOUD_CUBE.atoms)

    qapp = QApplication.instance() or QApplication(sys.argv)

    window = QWidget()
    window.setWindowTitle("IawMolShow")
    main_layout = QVBoxLayout()
    window.setLayout(main_layout)
    APP_CANVAS = scene.SceneCanvas(keys='interactive', show=False
                                , size=(1200, 1000)
                                , resizable = True
                                , config=dict(samples=4)
                                , vsync=True, app='pyqt6') 
    APP_VIEW = APP_CANVAS.central_widget.add_view()
    APP_VIEW.camera = scene.cameras.ArcballCamera(fov=45, center=(0, 0, 0))
    APP_CANVAS.bgcolor = apply_theme["background_color"]
    APP_VIEW.camera.aspect = 1.0
    APP_VIEW.camera.scale_factor = 1.0
    if bonds == None or len(bonds) == 0:        # 补丁, 防止原子间不能成键的情况
        ECLOUD_BONDS = []
    else:
        ECLOUD_BONDS = init_bonds(bonds, ECLOUD_CUBE.coor * BOHR_TO_A)
    ECLOUD_BALLS = init_atoms(ECLOUD_CUBE.coor * BOHR_TO_A, colors, radii)

    for i in ECLOUD_BONDS:
        APP_VIEW.add(i)
    for i in ECLOUD_BALLS:
        APP_VIEW.add(i)

    end_time2 = time.perf_counter()
    rp("Info\\[iaw]>: Build atoms and bonds time: {:.4f} seconds.".format(end_time2 - end_time1))

    ECLOUD_SCATTER = Markers()
    APP_VIEW.add(ECLOUD_SCATTER)
    init_cube_scatter(scale= ECLOUD_SMOOTH_FACTOR, smooth_method=ECLOUD_SMOOTH_METHOD)
    

    ECLOUD_MESH = Mesh(
        color=apply_theme["ecloud_mesh_color"],       
        shading=apply_theme["elound_shading"],
        mode='triangles'
    )
 
    APP_VIEW.add(ECLOUD_MESH)

    end_time3 = time.perf_counter()
    rp("Info\\[iaw]>: Initialize ecloud visuals time: {:.4f} seconds.".format(end_time3 - end_time2))

    # 添加画布和控制面板到主窗口
    panel = ControlPanel()
    main_layout.addWidget(panel)
    main_layout.addWidget(APP_CANVAS.native)
    window.show()

    sys.exit(qapp.exec())

if __name__ == "__main__":

    main()
