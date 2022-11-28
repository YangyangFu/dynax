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
    yangyangfu/diffbuildings-jax /bin/bash -c "cd /mnt/shared && python kalman_filter_example.py"