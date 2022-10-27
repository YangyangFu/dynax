docker run \
    --user=root \
    --detach=false \
    -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --rm \
    -v `pwd`:/mnt/shared \
    -i \
    -t \
    yangyangfu/diffbuilding-jl /bin/bash -c "cd /mnt/shared && export JULIA_NUM_THREADS=1 && julia simple_RC.jl"