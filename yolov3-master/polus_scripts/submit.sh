bsub -o job_stdout.txt \
    -e job_stderr.txt -W 00:01 -q normal \
    bash start_net.sh