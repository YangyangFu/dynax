open -a XQuartz && \
xhost + $(hostname) && \
docker run \
    --user=root \
    --detach=false \
    -e DISPLAY=$(hostname):0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --rm \
    -v `pwd`:/mnt/shared \
    -i \
    -t \
    yangyangfu/diffbuildings-jax /bin/bash -c "cd /mnt/shared && XLA_FLAGS=\"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1\" python parameter_inference.py"
