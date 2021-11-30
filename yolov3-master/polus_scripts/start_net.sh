echo $(pwd)
cd ..; python train.py --batch-size 1 --weights models/yolov3.pt > >(tee logfile) 2>&1