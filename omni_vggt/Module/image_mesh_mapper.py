import os
import cv2
import torch
import trimesh
import numpy as np
from tqdm import tqdm
from typing import List
from shutil import rmtree, copyfile

from camera_control.Method.data import toNumpy
from camera_control.Module.camera import Camera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from omni_vggt.Method.sample import sampleFibonacciSpherePoints


class ImageMeshMapper(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def createCameraList(
        mesh: trimesh.Trimesh,
        camera_num: int = 6,
        camera_dist: float = 2.5,
        width: int = 518,
        height: int = 518,
        fx: float = 500.0,
        fy: float = 500.0,
        dtype = torch.float32,
        device: str = 'cuda:0',
    ) -> List[Camera]:
        """
        创建围绕mesh均匀分布的相机和深度数据

        Args:
            mesh: 输入的三角网格
            camera_num: 相机数量
            camera_dist: 相机距离mesh中心的距离
            width: 图像宽度
            height: 图像高度
            fx: 焦距x
            fy: 焦距y

        Returns:
            camera_list: 相机列表
        """
        # 计算mesh的bbox center
        bbox = mesh.bounds  # shape: (2, 3), [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        bbox_center = (bbox[0] + bbox[1]) / 2.0

        # 使用Fibonacci球面采样生成均匀分布的相机位置
        camera_positions = sampleFibonacciSpherePoints(
            camera_num, camera_dist, bbox_center
        )[..., [2, 0, 1]]

        # 创建相机列表
        camera_list = []
        for i in range(camera_num):
            camera = Camera(
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                pos=camera_positions[i],
                look_at=bbox_center,
                up=[0, 1, 0],
                dtype=dtype,
                device=device,
            )
            camera_list.append(camera)

        return camera_list

    @staticmethod
    def sampleMeshCameraAndDepthData(
        mesh: trimesh.Trimesh,
        camera_num: int = 6,
        camera_dist: float = 2.5,
        width: int = 518,
        height: int = 518,
        fx: float = 500.0,
        fy: float = 500.0,
        dtype = torch.float32,
        device: str = 'cuda:0',
    ) -> tuple:
        camera_list = ImageMeshMapper.createCameraList(
            mesh=mesh,
            camera_num=camera_num,
            camera_dist=camera_dist,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            dtype=dtype,
            device=device,
        )

        depth_list = []
        depth_image_list = []

        for camera in camera_list:
            render_dict = NVDiffRastRenderer.renderDepth(
                mesh=mesh,
                camera=camera,
            )

            depth = render_dict['depth']
            depth_image = render_dict['image']

            depth_list.append(depth)
            depth_image_list.append(depth_image)

        return camera_list, depth_list, depth_image_list

    @staticmethod
    def createOmniVGGTDataFolder(
        image_file_path: str,
        mesh: trimesh.Trimesh,
        camera_list: List[Camera],
        save_data_folder_path: str,
    ) -> bool:
        if not os.path.exists(image_file_path):
            print('[ERROR][ImageMeshMapper::createOmniVGGTDataFolder]')
            print('\t image file not exist!')
            print('\t image_file_path:', image_file_path)
            return False

        if os.path.exists(save_data_folder_path):
            rmtree(save_data_folder_path)

        camera_folder_path = save_data_folder_path + 'cameras/'
        image_folder_path = save_data_folder_path + 'images/'
        depth_folder_path = save_data_folder_path + 'depths/'
        os.makedirs(camera_folder_path, exist_ok=True)
        os.makedirs(image_folder_path, exist_ok=True)
        os.makedirs(depth_folder_path, exist_ok=True)

        depth_vis_folder_path = save_data_folder_path + 'depths_vis/'
        os.makedirs(depth_vis_folder_path, exist_ok=True)

        print('[INFO][ImageMeshMapper::createOmniVGGTDataFolder]')
        print('\t start create omnivggt data folder...')
        for i, camera in tqdm(enumerate(camera_list)):
            render_image_dict = NVDiffRastRenderer.renderTexture(
                mesh=mesh,
                camera=camera,
            )

            render_depth_dict = NVDiffRastRenderer.renderDepth(
                mesh=mesh,
                camera=camera,
            )

            world2camera = toNumpy(camera.world2camera)
            intrinsic = toNumpy(camera.intrinsic)
            with open(camera_folder_path + str(i) + '.txt', 'w') as f:
                for j in range(3):
                    f.write(str(world2camera[j][0]) + '\t')
                    f.write(str(world2camera[j][1]) + '\t')
                    f.write(str(world2camera[j][2]) + '\t')
                    f.write(str(world2camera[j][3]) + '\n')
                for j in range(3):
                    f.write(str(intrinsic[j][0]) + '\t')
                    f.write(str(intrinsic[j][1]) + '\t')
                    f.write(str(intrinsic[j][2]) + '\n')

            image = render_image_dict['image']
            depth = render_depth_dict['depth']
            depth_image = render_depth_dict['image']

            cv2.imwrite(image_folder_path + str(i) + '.png', image)
            np.save(depth_folder_path + str(i) + '.npy', toNumpy(depth))
            cv2.imwrite(depth_vis_folder_path + str(i) + '.png', depth_image)

        copyfile(image_file_path, image_folder_path + 'target.png')
        return True
