export CUDA_VISIBLE_DEVICES=4,5,6,7

# Directory (on the DGX server): /data/ai/HE/moco_histology 

# Description: Train MoCo on histological images.

# An example of input file in TCGA_jpg_list.txt
#/bricks/brick1/TCIA/analysis/level1_patch_512x/trn/SARC/682448.jpg
#/bricks/brick1/TCIA/analysis/level1_patch_512x/trn/KIRP/1164439.jpg
#/bricks/brick1/TCIA/analysis/level1_patch_512x/trn/SKCM/761951.jpg
#/bricks/brick1/TCIA/analysis/level1_patch_512x/trn/UCEC/800594.jpg
#/bricks/brick1/TCIA/analysis/level1_patch_512x/trn/HNSC/535803.jpg
#/bricks/brick1/TCIA/analysis/level1_patch_512x/trn/HNSC/1117172.jpg
#/bricks/brick1/TCIA/analysis/level1_patch_512x/trn/LIHC/616975.jpg
#/bricks/brick1/TCIA/analysis/level1_patch_512x/trn/LUSC/180025.jpg
#/bricks/brick1/TCIA/analysis/level1_patch_512x/trn/SARC/1228653.jpg
#/bricks/brick1/TCIA/analysis/level1_patch_512x/trn/LGG/937093.jpg


/opt/software/install/miniconda37/bin/python moco/main_moco_he.py \
  -a resnet50w2 -j 16 -p 100 \
  --lr 0.015 --batch-size 128 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 TCGA_jpg_list.txt
  
