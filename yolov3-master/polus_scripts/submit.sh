bsub -o job_stdout.txt \
    -e job_stderr.txt -W 00:01 -q normal -gpu "num=1:mode=exclusive_process" \
    bash start_net.sh