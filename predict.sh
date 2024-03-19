work_path="./exp/animal/bce_logits"

data_path="/mount/ccai_nas/yunzhu/Animal_Kingdom/action_recognition/dataset/video/"
python ./testone.py \
  --cfg $work_path/config.yaml \
  --video_path $data_path/"AAACXZTV.mp4"
