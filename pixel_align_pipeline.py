import sys
sys.path.append('../camera-control')

import os
import cv2
import torch
import pickle
import numpy as np

from camera_control.Method.io import loadMeshFile
from camera_control.Module.camera import Camera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from omni_vggt.Module.image_mesh_mapper import ImageMeshMapper
from omni_vggt.Module.detector import Detector

def toGPU(pred, device):
    """
    Move all tensors in pred dict to specified device.
    """
    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            pred[k] = v.to(device)
    return pred

def save_predictions(pred, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pred, f)

def load_predictions(path):
    with open(path, "rb") as f:
        return pickle.load(f)

if __name__ == '__main__':
    shape_id = 'nezha'

    home = os.environ['HOME']
    model_file_path = home + '/chLi/Model/OmniVGGT/OmniVGGT.safetensors'
    image_file_path = home + "/chLi/Dataset/MM/Match/" + shape_id + "/" + shape_id + ".png"
    mesh_file_path = home + "/chLi/Dataset/MM/Match/" + shape_id + "/" + shape_id + ".glb"
    save_data_folder_path = home + "/chLi/Dataset/MM/Match/" + shape_id + "/omnivggt/"
    camera_num = 24
    camera_dist = 2.5
    device = 'cuda:0'

    tmp_output_pkl = "./output/tmp_" + shape_id + ".pkl"

    image = cv2.imread(image_file_path)
    mesh = loadMeshFile(mesh_file_path)

    if not os.path.exists(tmp_output_pkl):
        ImageMeshMapper.createOmniVGGTDataFolder(
            image_file_path,
            mesh,
            save_data_folder_path,
            camera_num=camera_num,
            camera_dist=camera_dist,
            device=device,
        )

        detector = Detector(model_file_path, device)
        predictions = detector.detect(
            image_folder_path=save_data_folder_path + 'images/',
            camera_folder_path=save_data_folder_path + 'cameras/',
            depth_folder_path=save_data_folder_path + 'depths/',
        )

        save_predictions(predictions, tmp_output_pkl)
        print(f"Predictions saved to {tmp_output_pkl}")
    else:
        predictions = load_predictions(tmp_output_pkl)
        predictions = toGPU(predictions, device)
        print(f"Loaded predictions from {tmp_output_pkl} and moved tensor data to {device}")

    for key, value in predictions.items():
        try:
            print(key, value.shape)
        except:
            pass

    depth = predictions['depth'][-1]
    depth_conf = predictions['depth_conf'][-1]
    extrinsic = predictions['extrinsic'][-1]
    intrinsic = predictions['intrinsic'][-1]
    world_points = predictions['world_points_from_depth'][-1]

    camera_0_file_path = save_data_folder_path + 'cameras/0.txt'

    gt_camera = Camera(
        width=image.shape[1],
        height=image.shape[0],
        device=device,
    )
    gt_camera.loadVGGTCameraFile(camera_0_file_path)

    print('gt camera 0:')
    print(gt_camera.camera2worldCV)

    camera_0_extrinsic = predictions['extrinsic'][0]
    camera_0_intrinsic = predictions['intrinsic'][0]

    camera2world_cv = np.eye(4)
    camera2world_cv[:3, :] = camera_0_extrinsic

    print('predict camera 0:')
    print(camera2world_cv)

    camera = Camera(
        width=image.shape[1],
        height=image.shape[0],
        fx=camera_0_intrinsic[0][0],
        fy=camera_0_intrinsic[1][1],
        cx=camera_0_intrinsic[0][2],
        cy=camera_0_intrinsic[1][2],
        device=device,
    )
    camera.setWorld2CameraByCamera2WorldCV(camera2world_cv)

    print('loaded camera 0:')
    print(camera.camera2worldCV)

    render_dict = NVDiffRastRenderer.renderTexture(mesh, gt_camera)
    render_image = render_dict['image']

    concat_img = np.concatenate([image, render_image], axis=1)

    os.makedirs(save_data_folder_path + '/tmp/', exist_ok=True)
    save_concat_path = save_data_folder_path + '/tmp/image_and_render_concat.png'
    cv2.imwrite(save_concat_path, concat_img)
    print(f"Concatenated image saved to {save_concat_path}")
