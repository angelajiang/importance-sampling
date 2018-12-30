set -x

EXP_NAME=$1

BASE_DIR="/proj/BigLearning/ahjiang/output/mnist/importance_sampling/"
OUTPUT_DIR=$BASE_DIR"/"$EXP_NAME"_sb"
mkdir $BASE_DIR
mkdir $OUTPUT_DIR

git rev-parse HEAD &> $OUTPUT_DIR/sha

python -u importance_sampling.py small_cnn \
   sb \
   sb \
   unweighted \
   mnist \
   $OUTPUT_DIR \
   --save_idxs_hist \
   --hyperparams 'batch_size=i128;lr=f0.003;lr_reductions=I10000' \
   --train_for 3000 \
   --validate_every 100 | tee $OUTPUT_DIR"/selectivity.txt"

