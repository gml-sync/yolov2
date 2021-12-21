rel_path=$(dirname "$0")                # relative
abs_path=$(cd "$rel_path" && pwd)       # absolutized and normalized

cd "$rel_path" && bsub -o job_stdout.txt \
    -e job_stderr.txt -W 00:02 -q normal -gpu "num=1:mode=exclusive_process" \
    bash train_restore.sh 