echo $(pwd)
cd ..; python train.py --weights models/yolov3.pt > >(tee logfile) 2>&1