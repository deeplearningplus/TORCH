export CUDA_VISIBLE_DEVICES=4,5,6,7

# Directory (on my DGX server): /data/ai/HE/moco_cytology

# Description: Train MoCo on cytological images.

# Example of input file in TJMUCH_Cellular_HE.txt:
#/data/ai/HE/TJMUCH_Cellular_HE/F201808317_001_800_2050.jpg
#/data/ai/HE/TJMUCH_Cellular_HE/509441_001_1050_1050.jpg
#/data/ai/HE/TJMUCH_Cellular_HE/F201805834_001_1800_1800.jpg
#/data/ai/HE/TJMUCH_Cellular_HE/F201806458_001_1800_800.jpg
#/data/ai/HE/TJMUCH_Cellular_HE/F201807973_001_800_2550.jpg
#/data/ai/HE/TJMUCH_Cellular_HE/S1800260_001_1050_50.jpg
#/data/ai/HE/TJMUCH_Cellular_HE/F201803393_001_300_300.jpg
#/data/ai/HE/TJMUCH_Cellular_HE/F201902746_001_800_2300.jpg
#/data/ai/HE/TJMUCH_Cellular_HE/HZ70564_001_300_2050.jpg
#/data/ai/HE/TJMUCH_Cellular_HE/S2004162_001_1050_2050.jpg

/opt/software/install/miniconda37/bin/python moco/main_moco_he.py \
  -a resnet50w2 -j 8 -p 100 \
  --lr 0.015 --batch-size 128 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ../TJMUCH_Cellular_HE.txt
