OMNIVGGT_DATA_FOLDER=$HOME/chLi/Dataset/MM/Match/nezha/omnivggt

python inference.py \
  --image_folder $OMNIVGGT_DATA_FOLDER/images/ \
  --camera_folder $OMNIVGGT_DATA_FOLDER/cameras/ \
  --depth_folder $OMNIVGGT_DATA_FOLDER/depths/ \
  --target_size 518 \
  --use_point_map \
  --conf_threshold 80.0 \
  --port 8080 \
  --save_glb \
  --background_mode
