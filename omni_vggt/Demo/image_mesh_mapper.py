import sys
sys.path.append('../camera-control')

import os

from camera_control.Method.io import loadMeshFile

from omni_vggt.Module.image_mesh_mapper import ImageMeshMapper


def demo() -> bool:
    home = os.environ['HOME']
    image_file_path = home + "/chLi/Dataset/MM/Match/nezha/nezha.png"
    mesh_file_path = home + "/chLi/Dataset/MM/Match/nezha/nezha.glb"
    save_data_folder_path = home + "/chLi/Dataset/MM/Match/nezha/omnivggt/"
    camera_num = 12
    camera_dist = 2.5
    device = 'cuda:0'

    mesh = loadMeshFile(mesh_file_path)

    camera_list = ImageMeshMapper.createCameraList(
        mesh,
        camera_num,
        camera_dist,
        device=device,
    )

    ImageMeshMapper.createOmniVGGTDataFolder(
        image_file_path,
        mesh,
        camera_list,
        save_data_folder_path,
    )

    return True
