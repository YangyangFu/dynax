docker run^
    --user=root^
    --detach=false^
    -e DISPLAY=${DISPLAY}^
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw^
    --rm^
    -v %CD%:/mnt/shared^
    -i^
    -t^
    yangyangfu/diffbuildings-jax-cpu /bin/bash -c "cd /mnt/shared && python ddqn-rc-dyna-lssm.py"