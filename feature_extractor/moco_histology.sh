export CUDA_VISIBLE_DEVICES=4,5,6,7

# Directory (on the DGX server): /data/ai/HE/moco_histology 

/opt/software/install/miniconda37/bin/python moco/main_moco_he.py \
  -a resnet50w2 -j 16 -p 100 \
  --lr 0.015 --batch-size 128 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 TCGA_jpg_list.txt
  
