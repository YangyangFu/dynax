HOST = yangyangfu

# define image names
IMAGE_NAME = diffbuildings
TAG_CPU = cpu
TAG_GPU = gpu
TAG_JL = jl
TAG_TAICHI = taichi
TAG_JAX = jax 

# some dockerfile
DOCKERFILE_CPU = Dockerfile_CPU
DOCKERFILE_GPU = Dockerfile_GPU
DOCKERFILE_JL = Dockerfile.jl
DOCKERFILE_TAICHI = Dockerfile.taichi
DOCKERFILE_JAX = Dockerfile.jax

#build:
#	docker build --build-arg CONDA_VERSION=${CONDA_VERSION},CONDA_MD5=${CONDA_MD5} --no-cache --rm -t ${IMA_NAME} .
build_cpu:
	docker build -f ${DOCKERFILE_CPU} --no-cache --rm -t ${HOST}/${IMAGE_NAME}-taichi:${TAG_CPU} .

build_gpu:
	docker build -f ${DOCKERFILE_GPU} --no-cache --rm -t ${HOST}/${IMAGE_NAME}-taichi:${TAG_GPU} .

build_jl: 
	docker build -f ${DOCKERFILE_JL} --no-cache --rm -t ${HOST}/${IMAGE_NAME}-${TAG_JL} .

build_taichi:
	docker build -f ${DOCKERFILE_TAICHI} --no-cache --rm -t ${HOST}/${IMAGE_NAME}-${TAG_TAICHI} .

build_jax:
	docker build -f ${DOCKERFILE_JAX} --no-cache --rm -t ${HOST}/${IMAGE_NAME}-${TAG_JAX} .