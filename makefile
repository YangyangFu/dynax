HOST = yangyangfu

# define image names
IMAGE_NAME = diffbuilding_taichi
TAG_CPU = cpu
TAG_GPU = gpu

# some dockerfile
DOCKERFILE_CPU = Dockerfile_CPU
DOCKERFILE_GPU = Dockerfile_GPU

#build:
#	docker build --build-arg CONDA_VERSION=${CONDA_VERSION},CONDA_MD5=${CONDA_MD5} --no-cache --rm -t ${IMA_NAME} .
build_cpu:
	docker build -f ${DOCKERFILE_CPU} --no-cache --rm -t ${HOST}/${IMAGE_NAME}:${TAG_CPU} .

build_gpu:
	docker build -f ${DOCKERFILE_GPU} --no-cache --rm -t ${HOST}/${IMAGE_NAME}:${TAG_GPU} .
