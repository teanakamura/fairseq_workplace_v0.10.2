CURRENT_DIR=`pwd`
SCRIPT_DIR=`dirname $0`
cd $SCRIPT_DIR
EXEC_FILE_PATH='../../fairseq/fairseq_cli/'
DATA_DIR='../data-bin/cnndm_small/'
SAVE_DIR='../checkpoints/cnndm_small/insertion_transformer/'

CUDA_VISIBLE_DEVICES=2,3 \
   python ${EXEC_FILE_PATH}interactive.py ${DATA_DIR} \
   --path ${SAVE_DIR} \
   --batch_size 128
   --beam 5
