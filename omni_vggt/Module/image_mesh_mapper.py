import os
import cv2
import torch
import trimesh
import numpy as np
from tqdm import trange
from shutil import rmtree, copyfile

from camera_control.Method.data import toNumpy
from camera_control.Method.sample import sampleCamera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer


class ImageMeshMapper(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def createOmniVGGTDataFolder(
        image_file_path: str,
        mesh: trimesh.Trimesh,
        save_data_folder_path: str,
        camera_num: int = 6,
        camera_dist: float = 2.5,
        width: int = 518,
        height: int = 518,
        fx: float = 500.0,
        fy: float = 500.0,
        dtype = torch.float32,
        device: str = 'cuda:0',
    ) -> bool:
        if not os.path.exists(image_file_path):
            print('[ERROR][ImageMeshMapper::createOmniVGGTDataFolder]')
            print('\t image file not exist!')
            print('\t image_file_path:', image_file_path)
            return False

        camera_list = sampleCamera(
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

        first_camera = camera_list[0]
        world2cameraCV_global = toNumpy(first_camera.world2cameraCV, np.float32)
        camera2worldCV_global = toNumpy(first_camera.camera2worldCV, np.float32)

        with open(camera_folder_path + 'c2wCV.txt', 'w') as f:
            for j in range(3):
                f.write(str(camera2worldCV_global[j][0]) + '\t')
                f.write(str(camera2worldCV_global[j][1]) + '\t')
                f.write(str(camera2worldCV_global[j][2]) + '\t')
                f.write(str(camera2worldCV_global[j][3]) + '\n')

        print('[INFO][ImageMeshMapper::createOmniVGGTDataFolder]')
        print('\t start create omnivggt data folder...')
        for i in trange(len(camera_list)):
            camera = camera_list[i]

            render_image_dict = NVDiffRastRenderer.renderTexture(
                mesh=mesh,
                camera=camera,
            )

            render_depth_dict = NVDiffRastRenderer.renderDepth(
                mesh=mesh,
                camera=camera,
            )

            camera2world = toNumpy(camera.camera2worldCV, np.float32) @ world2cameraCV_global
            intrinsic = toNumpy(camera.intrinsic, np.float32)
            with open(camera_folder_path + str(i) + '.txt', 'w') as f:
                for j in range(3):
                    f.write(str(camera2world[j][0]) + '\t')
                    f.write(str(camera2world[j][1]) + '\t')
                    f.write(str(camera2world[j][2]) + '\t')
                    f.write(str(camera2world[j][3]) + '\n')
                for j in range(3):
                    f.write(str(intrinsic[j][0]) + '\t')
                    f.write(str(intrinsic[j][1]) + '\t')
                    f.write(str(intrinsic[j][2]) + '\n')

            image = render_image_dict['image']
            depth = render_depth_dict['depth']
            depth_image = render_depth_dict['image']

            depth = torch.where(depth == 0, 1e11, depth)

            cv2.imwrite(image_folder_path + str(i) + '.png', image)
            np.save(depth_folder_path + str(i) + '.npy', toNumpy(depth, np.float32))
            cv2.imwrite(depth_vis_folder_path + str(i) + '.png', depth_image)

        copyfile(image_file_path, image_folder_path + 'target.png')
        return True
