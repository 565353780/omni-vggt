OMNIVGGT_DATA_FOLDER=$HOME/chLi/Dataset/MM/Match/nezha/omnivggt

python inference.py \
  --image_folder $OMNIVGGT_DATA_FOLDER/images/ \
  --camera_folder $OMNIVGGT_DATA_FOLDER/cameras/ \
  --depth_folder $OMNIVGGT_DATA_FOLDER/depths/ \
  --target_size 518 \
  --conf_threshold 25.0 \
  --port 8080 \
  --background_mode \
  --save_glb
