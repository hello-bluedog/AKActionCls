NUM_SHARDS=1
NUM_GPUS=2
BATCH_SIZE=32
BASE_LR=1e-5
work_path=./exp/animal/umtvit_samp_32
PYTHONPATH=$PYTHONPATH:./slowfast \
python tools/run_net.py \
  --init_method tcp://localhost:10124 \
  --cfg $work_path/config.yaml \
  --num_shards $NUM_SHARDS \
  DATA.PATH_TO_DATA_DIR  /mount/ccai_nas/yunzhu/Animal_Kingdom/action_recognition/annotation/ \
  DATA.PATH_PREFIX  /mount/ccai_nas/yunzhu/Animal_Kingdom/action_recognition/dataset/image/ \
  DATA.PATH_LABEL_SEPARATOR " " \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 100 \
  TRAIN.BATCH_SIZE $BATCH_SIZE \
  TRAIN.SAVE_LATEST False \
  NUM_GPUS $NUM_GPUS \
  NUM_SHARDS $NUM_SHARDS \
  SOLVER.MAX_EPOCH 55 \
  SOLVER.BASE_LR $BASE_LR \
  SOLVER.BASE_LR_SCALE_NUM_SHARDS False \
  SOLVER.WARMUP_EPOCHS 5. \
  TRAIN.ENABLE False \
  TEST.NUM_ENSEMBLE_VIEWS 4 \
  TEST.NUM_SPATIAL_CROPS 3 \
  TEST.TEST_BEST True \
  TEST.ADD_SOFTMAX False \
  TEST.BATCH_SIZE 128 \
  TEST.CHECKPOINT_FILE_PATH "/mount/ccai_nas/yunzhu/Animal_Kingdom/output/umtvit_samp_32/checkpoints/checkpoint_epoch_00140.pyth"
  RNG_SEED 6666 \
  OUTPUT_DIR "/mount/ccai_nas/yunzhu/Animal_Kingdom/output/umtvit_samp_32/"
