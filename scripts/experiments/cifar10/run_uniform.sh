set -x

EXP_NAME=$1

BASE_DIR="/proj/BigLearning/ahjiang/output/cifar10/importance_sampling/"
OUTPUT_DIR=$BASE_DIR"/"$EXP_NAME"_uniform"
mkdir $BASE_DIR
mkdir $OUTPUT_DIR

git rev-parse HEAD &> $OUTPUT_DIR/sha

python -u importance_sampling.py \
  wide_resnet_28_2 \
  oracle-gnorm \
  uniform \
  unweighted \
  cifar10 \
  $OUTPUT_DIR \
  --save_idxs_hist \
  --hyperparams 'batch_size=i128;forward_batch_size=i128;lr=f0.1;momentum=f0.9;opt=ssgd;lr_changes=I20000!40000;lr_targets=F0.02!0.004' \
  --train_for 50000 \
  --validate_every 100 | tee $OUTPUT_DIR"/selectivity.txt"
