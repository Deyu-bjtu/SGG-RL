# export PYTHONPATH=/mnt/hdd1/zhanghaonan/code/code_sgg/lib/apex:/mnt/hdd1/zhanghaonan/code/code_sgg/lib/cocoapi:/mnt/hdd1/zhanghaonan/code/code_sgg/PE-Net/Scene-Graph-Benchmark.pytorch-master:$PYTHONPATH

# export CUDA_VISIBLE_DEVICES=6
# export NUM_GUP=1
# echo "TRAINING Predcls"

# MODEL_NAME='PE-NET_PredCls'
# mkdir ./checkpoints/${MODEL_NAME}/
# cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
# cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
# cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
# cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/loss.py ./checkpoints/${MODEL_NAME}/
# cp ./scripts/train.sh ./checkpoints/${MODEL_NAME}/
# cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/relation_head.py ./checkpoints/${MODEL_NAME}

# python3 \
#   tools/relation_train_net.py \
#   --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
#   MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
#   MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
#   MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
#   MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
#   DTYPE "float32" \
#   SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH $NUM_GUP \
#   SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-3 \
#   SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
#   MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
#   SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 30000 \
#   SOLVER.CHECKPOINT_PERIOD 30000 GLOVE_DIR ./datasets/vg/ \
#   MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
#   OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
#   SOLVER.PRE_VAL False \
#   SOLVER.GRAD_NORM_CLIP 5.0;

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

POSSIBLE_PATHS=(
    "$HOME/anaconda3"
    "/opt/anaconda3"
)

# 搜索并 source conda.sh
for path in "${POSSIBLE_PATHS[@]}"; do
    if [[ -f "$path/etc/profile.d/conda.sh" ]]; then
        source "$path/etc/profile.d/conda.sh"
        echo "Conda environment sourced from $path"
        break
    fi
done

conda activate sgg_benchmark

target_free_memory=20000
while true; do
    # 仅获取第一个GPU的显存总量和已使用量
    memory_info=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0)
    
    # 计算空余显存
    total_memory=$(echo $memory_info | cut -d ',' -f 1 | tr -d '[:space:]')
    used_memory=$(echo $memory_info | cut -d ',' -f 2 | tr -d '[:space:]')
    free_memory=$((total_memory - used_memory))

    # 检查空余显存是否达到目标
    if [ "$free_memory" -gt "$target_free_memory" ]; then
        break
    else
        sleep 120
    fi
done

export CUDA_LAUNCH_BLOCKING=1

cuda_device=0,1,2,3
IFS=',' read -r -a array <<< "$cuda_device"
NUM_GUP=${#array[@]}

PER_BATCH_SIZE=2  # if PER_BATCH_SIZE=1 ==> BATCH_SIZE=4 ==> SOLVER.MAX_ITER=60000*2
MAX_ITER=120000   # if PER_BATCH_SIZE=2 ==> BATCH_SIZE=8 ==> SOLVER.MAX_ITER=60000
MODEL_NAME='EntityTrans_v2'

PRETRAINED_DETECTOR_CKPT="/data/sdc/pretrain_model/pretrained_faster_rcnn/model_final.pth"  # "/data/sdb/pretrain_ckpt/pretrained_faster_rcnn/model_final.pth"
GLOVE_DIR="/data/sdc/pretrain_model/glove"
ZEROSHOT_TYPE="None"

CUDA_VISIBLE_DEVICES=$cuda_device python -m torch.distributed.launch --nproc_per_node=$NUM_GUP --master_addr="127.0.0.1" --master_port=1647 tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR $MODEL_NAME \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH $(expr $NUM_GUP \* $PER_BATCH_SIZE) TEST.IMS_PER_BATCH $NUM_GUP \
  SOLVER.MAX_ITER $MAX_ITER SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  SOLVER.PRE_VAL False \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 20000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  MODEL.PRETRAINED_DETECTOR_CKPT $PRETRAINED_DETECTOR_CKPT \
  GLOVE_DIR $GLOVE_DIR \
  OUTPUT_DIR outputs/${MODEL_NAME} \
  SOLVER.GRAD_NORM_CLIP 5.0 \
  TEST.ALLOW_LOAD_FROM_CACHE False \
  SOLVER.ZEROSHOT_MODE $ZEROSHOT_TYPE \
  MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER 4 \
  ${@:1}
