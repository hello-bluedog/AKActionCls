NUM_SHARDS=1
NUM_GPUS=4
BATCH_SIZE=256
BASE_LR=1e-5
work_path="./exp/animal/mamba_t"
PYTHONPATH=$PYTHONPATH:./slowfast \
python -X faulthandler tools/run_net.py \
  --init_method tcp://localhost:10123 \
  --cfg $work_path/config.yaml \
  --num_shards $NUM_SHARDS \
  DATA.PATH_TO_DATA_DIR /mount/ccai_nas/yunzhu/Animal_Kingdom/action_recognition/annotation/ \
  DATA.PATH_PREFIX /mount/ccai_nas/yunzhu/Animal_Kingdom/action_recognition/dataset/image/ \
  DATA.PATH_LABEL_SEPARATOR " " \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 5 \
  TRAIN.BATCH_SIZE $BATCH_SIZE \
  TRAIN.SAVE_LATEST False \
  NUM_GPUS $NUM_GPUS \
  NUM_SHARDS $NUM_SHARDS \
  SOLVER.MAX_EPOCH 100 \
  SOLVER.BASE_LR $BASE_LR \
  SOLVER.BASE_LR_SCALE_NUM_SHARDS False \
  SOLVER.WARMUP_EPOCHS 5. \
  TEST.NUM_ENSEMBLE_VIEWS 4 \
  TEST.NUM_SPATIAL_CROPS 3 \
  TEST.TEST_BEST False \
  TEST.ADD_SOFTMAX False \
  TEST.BATCH_SIZE 64 \
  RNG_SEED 6666 \
  OUTPUT_DIR "/mount/ccai_nas/yunzhu/Animal_Kingdom/output/mamba_t"
