HOST = yangyangfu

# define image names
IMAGE_NAME = diffbuilding
TAG_CPU = cpu
TAG_GPU = gpu
TAG_JL = jl

# some dockerfile
DOCKERFILE_CPU = Dockerfile_CPU
DOCKERFILE_GPU = Dockerfile_GPU
DOCKERFILE_JL = Dockerfile.jl

#build:
#	docker build --build-arg CONDA_VERSION=${CONDA_VERSION},CONDA_MD5=${CONDA_MD5} --no-cache --rm -t ${IMA_NAME} .
build_cpu:
	docker build -f ${DOCKERFILE_CPU} --no-cache --rm -t ${HOST}/${IMAGE_NAME}-taichi:${TAG_CPU} .

build_gpu:
	docker build -f ${DOCKERFILE_GPU} --no-cache --rm -t ${HOST}/${IMAGE_NAME}-taichi:${TAG_GPU} .

build_jl: 
	docker build -f ${DOCKERFILE_JL} --no-cache --rm -t ${HOST}/${IMAGE_NAME}-${TAG_JL} .
