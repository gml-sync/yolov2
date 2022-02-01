rel_path=$(dirname "$0")                # relative
abs_path=$(cd "$rel_path" && pwd)       # absolutized and normalized

cd "$rel_path/.." && python train.py --weights models/yolov3.pt --data coco5k_ref.yaml > >(tee logfile) 2>&1