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

conda activate maskrcnn

export CUDA_LAUNCH_BLOCKING=1

target_free_memory=10000
cuda_device=0,1,2,3
first_cuda=$(echo "$cuda_device" | cut -d ',' -f 1)
IFS=',' read -r -a array <<< "$cuda_device"
NUM_GUP=${#array[@]}

while true; do
    # 仅获取第一个GPU的显存总量和已使用量
    memory_info=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$first_cuda")
    
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

# PER_BATCH_SIZE=4  # if PER_BATCH_SIZE=1 ==> BATCH_SIZE=4 ==> SOLVER.MAX_ITER=60000*2
# MAX_ITER=80000   # if PER_BATCH_SIZE=2 ==> BATCH_SIZE=8 ==> SOLVER.MAX_ITER=60000
PER_BATCH_SIZE=4
MAX_ITER=80000
BASE_LR=1e-3

MODEL_NAME="TransformerPredictor"  # Transformer_Relcenter, Motif_Relcenter, VCTree_Relcenter
AUXILIARY_MODULE="Multi_step_Denoise"

GLOVE_DIR="/data/sdc/pretrain_ckpt/glove"
PRETRAIN_PATH='/data/sdc/pretrain_ckpt/pretrained_faster_rcnn'
DATA_DIR="/data/sdc/SGG_data"

USE_GT_BOX=True
USE_GT_OBJECT_LABEL=True
PREDICT_USE_BIAS=False

DATASET_CHOICE="VG"
if [ "$DATASET_CHOICE" = "VG" ]; then
    SKIP_TEST=""
    CONFIG_FILE="configs/e2e_relation_X_101_32_8_FPN_1x.yaml"
    PRETRAINED_DETECTOR_CKPT=$PRETRAIN_PATH/model_final.pth  # "/data/sdb/pretrain_ckpt/pretrained_faster_rcnn/model_final.pth"
elif [ "$DATASET_CHOICE" = "GQA" ]; then
    SKIP_TEST=""
    CONFIG_FILE="configs/e2e_relation_X_101_32_8_FPN_1xGQA.yaml"
    PRETRAINED_DETECTOR_CKPT=$PRETRAIN_PATH/gqa_model_final_from_vg.pth  # "/data/sdb/pretrain_ckpt/pretrained_faster_rcnn/model_final.pth"
elif [ "$DATASET_CHOICE" = "OI_V4" ]; then
    SKIP_TEST="--skip-test"
    USE_GT_BOX=False
    USE_GT_OBJECT_LABEL=False
    CONFIG_FILE="configs/e2e_relation_X_101_32_8_FPN_1x_for_OIV4.yaml"
    PRETRAINED_DETECTOR_CKPT=$PRETRAIN_PATH/oiv4_det.pth  # "/data/sdb/pretrain_ckpt/pretrained_faster_rcnn/model_final.pth"
elif [ "$DATASET_CHOICE" = "OI_V6" ]; then
    SKIP_TEST="--skip-test"
    USE_GT_BOX=False
    USE_GT_OBJECT_LABEL=False
    CONFIG_FILE="configs/e2e_relation_X_101_32_8_FPN_1x_for_OIV6.yaml"
    PRETRAINED_DETECTOR_CKPT=$PRETRAIN_PATH/oiv6_det.pth  # "/data/sdb/pretrain_ckpt/pretrained_faster_rcnn/model_final.pth"
else
    echo "DATASET_CHOICE ValueError, must be 'VG', 'GQA', 'OI_V4', 'OI_V6'. "
    exit 1
fi

if [ "$USE_GT_BOX" = "True" ] && [ "$USE_GT_OBJECT_LABEL" = "True" ]; then
    mode="predcls"
elif [ "$USE_GT_BOX" = "True" ] && [ "$USE_GT_OBJECT_LABEL" = "False" ]; then
    mode="sgcls"
elif [ "$USE_GT_BOX" = "False" ] && [ "$USE_GT_OBJECT_LABEL" = "False" ]; then
    mode="sgdet"
else
    echo "Invalid combination of USE_GT_BOX and USE_GT_OBJECT_LABEL"
    exit 1
fi

if [[ $MODEL_NAME == *VCTree* ]] && [[ "$DATASET_CHOICE" == "VG" ]]; then
    CONTEXT_HIDDEN_DIM=1024
else
    CONTEXT_HIDDEN_DIM=512
fi

if [ "$PREDICT_USE_BIAS" = "True" ]; then
    OUTPUT_DIR=/data/sdc/checkpoints/SGG_Benchmark/${DATASET_CHOICE}/${AUXILIARY_MODULE}_v2/${mode}_step2
else
    OUTPUT_DIR=/data/sdc/checkpoints/SGG_Benchmark/${DATASET_CHOICE}/${AUXILIARY_MODULE}_v2/${mode}_wo_bias_step2
fi

if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi
cp maskrcnn_benchmark/modeling/roi_heads/relation_head/model_utils.py $OUTPUT_DIR
cp maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py $OUTPUT_DIR

PRETRAINED_DETECTOR_CKPT="/data/sdc/checkpoints/SGG_Benchmark/VG/Multi_step_Denoise_v2/predcls_wo_bias_step1/model_final.pth"

CUDA_VISIBLE_DEVICES=$cuda_device python -m torch.distributed.launch --nproc_per_node=$NUM_GUP --master_addr="127.0.0.1" --master_port=1643 tools/relation_train_net.py \
  --config-file $CONFIG_FILE $SKIP_TEST \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX $USE_GT_BOX \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL $USE_GT_OBJECT_LABEL \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS $PREDICT_USE_BIAS \
  MODEL.ROI_RELATION_HEAD.PREDICTOR $MODEL_NAME \
  MODEL.ROI_RELATION_HEAD.AUXILIARY_MODULE $AUXILIARY_MODULE \
  MODEL.ROI_RELATION_HEAD.TRAIN_STEP 2 \
  MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM $CONTEXT_HIDDEN_DIM \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH $(expr $NUM_GUP \* $PER_BATCH_SIZE) TEST.IMS_PER_BATCH $NUM_GUP \
  SOLVER.MAX_ITER $MAX_ITER SOLVER.BASE_LR $BASE_LR \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  SOLVER.PRE_VAL False \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 20000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  MODEL.PRETRAINED_DETECTOR_CKPT $PRETRAINED_DETECTOR_CKPT \
  SOLVER.DATASET_CHOICE $DATASET_CHOICE \
  DATASETS.DATA_DIR $DATA_DIR \
  GLOVE_DIR $GLOVE_DIR \
  OUTPUT_DIR $OUTPUT_DIR \
  SOLVER.GRAD_NORM_CLIP 5.0 \
  TEST.ALLOW_LOAD_FROM_CACHE False \
  MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER 3 \
  ${@:1} ;
