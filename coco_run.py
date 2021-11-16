import subprocess
import os
from tqdm import tqdm
import time

DATASETS_NAMES = [
        '010', '012', '014', 
        '016', '018', '020',
        '022', '025', '028',
        '030', '035', '040', 
        '045', '050', '055',
        '060', '065', '070', 
        '080', '090', '100', 
        'ref'
]

def runAll(project, weights, dataset_template, datasets=None):
    """Run all tests.

    Params:
        project - path to output, for example: "yolov5_runs/test_DD.MM.YYYY"
        weights - yolov5 weights, for example: "yolov5s.pt"
        dataset_template - template path to all datasets, for example: "../datasets/coco128_"
        datasets - names of datasets, every name will be substituted after `dataset_template`
    """

    """
    .---datasets---coco128_010
               |---coco128_012
               |--- ...
               |--- coco128_010.yaml
               └--- ...
    """
    
    if datasets is None:
        datasets = DATASETS_NAMES
    
    

    for name in tqdm(datasets):
        print( f"{dataset_template}{name}.yaml", os.getcwd() )
        args = f"python val.py --weights {weights} --data {dataset_template}{name}.yaml --task val --exist-ok --name {name} --project {project} --save-json --save-txt --save-conf"
        result = os.system(args)
        # with open('output.txt', 'ab') as f:
        #     f.write(result.stdout)
        sleep(10)

    
    zip_fname = f"{os.path.basename(project)}.zip"
    subprocess.run(f"zip {zip_fname} ./{project}/*/*.csv", shell=True)
    subprocess.run(f"zip {zip_fname} ./{project}/*/*.pkl", shell=True)
    subprocess.run(f"zip {zip_fname} ./{project}/*/labels/*.txt", shell=True)
    time.sleep(1)


def main():
    runAll('temp/test_11.10_128a', 'yolov5s.pt', "upload/datasets/coco128_", None)

main()