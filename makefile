HOST = yangyangfu

# define image names
IMAGE_NAME = dynax
TAG_JL = jl
TAG_JAX_CPU = jax-cpu
TAG_JAX_CUDA = jax-cuda 
TAG_TORCH_CPU = torch-cpu

# some dockerfile
DOCKERFILE_JL = Dockerfile.jl
DOCKERFILE_JAX = Dockerfile.jax
DOCKERFILE_JAX_CUDA = DockerfileCuda.jax
DOCKERFILE_TORCH = Dockerfile.torch

build_jl: 
	docker build -f ${DOCKERFILE_JL} --no-cache --rm -t ${HOST}/${IMAGE_NAME}-${TAG_JL} .

build_jax_cpu:
	docker build -f ${DOCKERFILE_JAX} --no-cache --rm -t ${HOST}/${IMAGE_NAME}-${TAG_JAX_CPU} .

build_jax_cuda:
	docker build -f ${DOCKERFILE_JAX_CUDA} --no-cache --rm -t ${HOST}/${IMAGE_NAME}-${TAG_JAX_CUDA} .

build_torch_cpu:
	docker build -f ${DOCKERFILE_TORCH} --no-cache --rm -t ${HOST}/${IMAGE_NAME}-${TAG_TORCH_CPU} .