import os
import torch
from typing import Optional
from safetensors.torch import load_file

from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri
from omnivggt.utils.misc import select_first_batch
from visual_util import load_images_and_cameras
from visual_util import (
    get_world_points_from_depth,
    load_images_and_cameras,
    predictions_to_glb,
)


class Detector(object):
    def __init__(
        self,
        model_file_path: Optional[str]=None,
        device: str = 'cuda:0',
    ) -> None:
        self.device = device

        self.model = OmniVGGT().to(self.device)

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][Detector::loadModel]')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            return False

        state_dict = load_file(model_file_path)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        return True

    def detect(
        self,
        image_folder_path: str,
        camera_folder_path: Optional[str]=None,
        depth_folder_path: Optional[str]=None,
    ) -> dict:
        images, extrinsics, intrinsics, depthmaps, masks, depth_indices, camera_indices = \
            load_images_and_cameras(
                image_folder=image_folder_path,
                camera_folder=camera_folder_path,
                depth_folder=depth_folder_path,
                target_size=518,
            )

        inputs = {
            'images': images.to(self.device),
            'extrinsics': extrinsics.to(self.device),
            'intrinsics': intrinsics.to(self.device),
            'depth': depthmaps.to(self.device),
            'mask': masks.to(self.device),
            'depth_gt_index': depth_indices,
            'camera_gt_index': camera_indices
        }

        with torch.no_grad():
            predictions = self.model(**inputs)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            images.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        predictions_0 = select_first_batch(predictions)
        get_world_points_from_depth(predictions_0)
        return predictions_0

    def saveAsGLB(
        self,
        predictions: dict,
        image_folder_path: str,
        save_glb_file_path: str,
    ) -> bool:
        print("Exporting scene to GLB...")
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=0.0,
            filter_by_frames='All',
            mask_white_bg=False,
            show_cam=True,
            mask_sky=False,
            target_dir=image_folder_path,
            prediction_mode="Predicted Depth",
        )
        glbscene.export(file_obj=save_glb_file_path)
        print(f"Saved GLB file to {save_glb_file_path}")
        return True
