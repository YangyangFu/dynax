FROM julia:1.8.2
LABEL Yangyang Fu

USER root

# add user 
RUN export uid=1000 gid=1000 && \
    mkdir -p /home/developer && \
    mkdir -p /etc/sudoers.d && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer && \
    chown ${uid}:${gid} -R /home/developer && \
    mkdir -m 1777 /tmp/.X11-unix

USER developer
ENV HOME /home/developer

# add julia packages
RUN julia -e 'using Pkg; Pkg.activate(); Pkg.instantiate(); \
    Pkg.add("ModelingToolkitStandardLibrary"); \
    Pkg.add("Lux"); \
    Pkg.add("Flux"); \
    Pkg.add("ForwardDiff"); \
    Pkg.add("Zygote"); \
    Pkg.add("OptimizationOptimisers"); \
    Pkg.add("ModelingToolkit"); \
    Pkg.add("OptimizationNLopt"); \
    Pkg.add("OptimizationEvolutionary"); \
    Pkg.add("DiffEqFlux"); \ 
    Pkg.add("DifferentialEquations"); \
    Pkg.add("Optimization"); \
    Pkg.add("OptimizationOptimJL"); \
    Pkg.add("Plots"); \
    Pkg.add("Random"); \
    using ModelingToolkitStandardLibrary; \
    using Lux; \
    using Flux; \
    using ForwardDiff; \
    using Zygote; \
    using OptimizationOptimisers; \
    using ModelingToolkit; \
    using OptimizationNLopt; \
    using OptimizationEvolutionary; \
    using DiffEqFlux; \
    using DifferentialEquations; \
    using Optimization; \
    using OptimizationOptimJL; \
    using Plots; \
    using Random'
