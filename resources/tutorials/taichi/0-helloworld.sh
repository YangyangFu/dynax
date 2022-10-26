open -a XQuartz && \
xhost + $(hostname) && \
docker run \
    --user=root \
    --detach=false \
    -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --rm \
    -v `pwd`:/mnt/shared \
    -i \
    -t \
    diffbuilding_taichi:cpu /bin/bash -c "cd /mnt/shared && python 0-helloworld.py"