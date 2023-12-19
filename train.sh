
./IRENE/train.py --train-file train.csv --val-file test.csv --outdir cyto-histo-layer3-irene-ckpt --cyto-pt-file cyto_pt_file_layer3 --histo-pt-file histo_pt_file_layer3 --mode both 
./IRENE/train.py --train-file train.csv --val-file test.csv --outdir cyto-layer3-irene-ckpt --cyto-pt-file cyto_pt_file_layer3 --histo-pt-file histo_pt_file_layer3 --mode cyto
./IRENE/train.py --train-file train.csv --val-file test.csv --outdir histo-layer3-irene-ckpt --cyto-pt-file cyto_pt_file_layer3 --histo-pt-file histo_pt_file_layer3 --mode histo


./models/train.py --model-name AttnMIL --train-file train.csv --val-file test.csv --outdir cyto-histo-layer3-AttnMIL-ckpt --cyto-pt-file cyto_pt_file_layer3 --histo-pt-file histo_pt_file_layer3 --mode both 
./models/train.py --model-name AttnMIL --train-file train.csv --val-file test.csv --outdir cyto-layer3-AttnMIL-ckpt --cyto-pt-file cyto_pt_file_layer3 --histo-pt-file histo_pt_file_layer3 --mode cyto
./models/train.py --model-name AttnMIL --train-file train.csv --val-file test.csv --outdir histo-layer3-AttnMIL-ckpt --cyto-pt-file cyto_pt_file_layer3 --histo-pt-file histo_pt_file_layer3 --mode histo 


./models/train.py --model-name WIT --train-file train.csv --val-file test.csv --outdir cyto-histo-layer3-WIT-ckpt --cyto-pt-file cyto_pt_file_layer3 --histo-pt-file histo_pt_file_layer3 --mode both 
./models/train.py --model-name WIT --train-file train.csv --val-file test.csv --outdir cyto-layer3-WIT-ckpt --cyto-pt-file cyto_pt_file_layer3 --histo-pt-file histo_pt_file_layer3 --mode cyto
./models/train.py --model-name WIT --train-file train.csv --val-file test.csv --outdir histo-layer3-WIT-ckpt --cyto-pt-file cyto_pt_file_layer3 --histo-pt-file histo_pt_file_layer3 --mode histo 


./models/train.py --model-name CLAM_MB --train-file train.csv --val-file test.csv --outdir cyto-histo-layer3-CLAM-MB-ckpt --cyto-pt-file cyto_pt_file_layer3 --histo-pt-file histo_pt_file_layer3 --mode both 
./models/train.py --model-name CLAM_MB --train-file train.csv --val-file test.csv --outdir cyto-layer3-CLAM-MB-ckpt --cyto-pt-file cyto_pt_file_layer3 --histo-pt-file histo_pt_file_layer3 --mode cyto
./models/train.py --model-name CLAM_MB --train-file train.csv --val-file test.csv --outdir histo-layer3-CLAM-MB-ckpt --cyto-pt-file cyto_pt_file_layer3 --histo-pt-file histo_pt_file_layer3 --mode histo 

