
ARG TRITON_VERSION=2.53.0dev
ARG TRITON_CONTAINER_VERSION=24.12dev
#ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.11-py3-min
ARG BASE_IMAGE=almalinux:8.9


############################################################################
##  Production stage: Create container with just inference server executable
############################################################################
FROM ${BASE_IMAGE}

ENV PIP_BREAK_SYSTEM_PACKAGES=1

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION

ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
LABEL com.nvidia.tritonserver.version="${TRITON_SERVER_VERSION}"

ENV PATH /opt/tritonserver/bin:${PATH}
# Remove once https://github.com/openucx/ucx/pull/9148 is available
# in the min container.
ENV UCX_MEM_EVENTS no

ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2
ENV TRITON_SERVER_GPU_ENABLED    1

# Create a user that can be used to run triton as
# non-root. Make sure that this user to given ID 1000. All server
# artifacts copied below are assign to this user.
ENV TRITON_SERVER_USER=triton-server
RUN userdel tensorrt-server > /dev/null 2>&1 || true \
      && userdel ubuntu > /dev/null 2>&1 || true \
      && if ! id -u $TRITON_SERVER_USER > /dev/null 2>&1 ; then \
          useradd $TRITON_SERVER_USER; \
        fi \
      && [ `id -u $TRITON_SERVER_USER` -eq 1000 ] \
      && [ `id -g $TRITON_SERVER_USER` -eq 1000 ]

# Common dependencies.
RUN yum install -y epel-release
RUN yum install --enablerepo=powertools -y \
        git \
        automake \
        gcc-toolset-9 \
        gcc-toolset-9-libstdc++-devel \
        cmake \
        gperf \
        re2-devel \
        libffi-devel \
        openssl-devel \
        libtool \
        libcurl-devel \
        libb64-devel \
        gperftools-devel \
        patchelf \
        wget \
        python3-pip \
        numactl-devel \
        libarchive-devel \
        openssl-devel \
        readline-devel

RUN scl enable gcc-toolset-9 bash


# Set TCMALLOC_RELEASE_RATE for users setting LD_PRELOAD with tcmalloc
ENV TCMALLOC_RELEASE_RATE 200

#ENV DCGM_VERSION 3.3.6
## Install DCGM. Steps from https://developer.nvidia.com/dcgm#Downloads
RUN dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf clean expire-cache \
    && dnf install -y datacenter-gpu-manager-3.3.6
#
## Install CUDA Toolkit
RUN dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf clean expire-cache \
    && dnf install -y cuda-toolkit-12-6
#
#RUN dnf -y module install nvidia-driver:open-dkms
RUN dnf -y install cudnn9-cuda-12

RUN dnf -y install libnccl libcusparselt0

# python3, python3-pip and some pip installs required for the python backend

# The python library version available for install via 'yum install python3.X-devel' does not
# match the version of python inside the RHEL base container. This means that python packages
# installed within the container will not be picked up by the python backend stub process pybind
# bindings. It must instead must be installed via pyenv.
ENV PYENV_ROOT=/opt/pyenv_build
RUN curl https://pyenv.run | bash
ENV PATH="${PYENV_ROOT}/bin:$PATH"
RUN eval "$(pyenv init -)"
RUN CONFIGURE_OPTS="--with-openssl=/usr/lib64" && pyenv install 3.12.3 \
    && cp ${PYENV_ROOT}/versions/3.12.3/lib/libpython3* /usr/lib64/

# RHEL image has several python versions. It's important
# to set the correct version, otherwise, packages that are
# pip installed will not be found during testing.
ENV PYVER=3.12.3 PYTHONPATH=/opt/python
RUN ln -sf ${PYENV_ROOT}/versions/${PYVER}* ${PYTHONPATH}
RUN ln -sf ${PYTHONPATH}/bin/python3 /usr/bin/python3
ENV PYBIN=${PYTHONPATH}/bin
ENV PYTHON_BIN_PATH=${PYBIN}/python${PYVER} PATH=${PYBIN}:${PATH}

RUN pip3 install --upgrade pip \
    && pip3 install --upgrade \
        wheel \
        setuptools \
        "numpy<2" \
        virtualenv

WORKDIR /opt/tritonserver
RUN rm -fr /opt/tritonserver/*
ENV NVIDIA_PRODUCT_NAME="Triton Server"
COPY docker/entrypoint.d/ /opt/nvidia/entrypoint.d/

ENV NVIDIA_BUILD_ID <unknown>
LABEL com.nvidia.build.id=<unknown>
LABEL com.nvidia.build.ref=

WORKDIR /opt
COPY --chown=1000:1000 build/install tritonserver

WORKDIR /opt/tritonserver
COPY --chown=1000:1000 NVIDIA_Deep_Learning_Container_License.pdf .

RUN find /opt/tritonserver/python -maxdepth 1 -type f -name \
    "tritonserver-*.whl" | xargs -I {} pip install --upgrade {}[all] && \
    find /opt/tritonserver/python -maxdepth 1 -type f -name \
    "tritonfrontend-*.whl" | xargs -I {} pip install --upgrade {}[all]

