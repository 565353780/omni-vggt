import sys
sys.path.append('../camera-control')

import os
import torch
import pickle

from camera_control.Method.io import loadMeshFile

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
            print(key, value)
