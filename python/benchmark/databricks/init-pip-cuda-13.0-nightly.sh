#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# set portion of path below after /dbfs/ to dbfs zip file location
SPARK_RAPIDS_ML_ZIP=/dbfs/path/to/spark-rapids-ml.zip
BENCHMARK_ZIP=/dbfs/path/to/benchmark.zip
# IMPORTANT: specify rapids fully 23.10.0 and not 23.10
# also, in general, RAPIDS_VERSION (python) fields should omit any leading 0 in month/minor field (i.e. 23.8.0 and not 23.08.0)
# while SPARK_RAPIDS_VERSION (jar) should have leading 0 in month/minor (e.g. 23.08.2 and not 23.8.2)
RAPIDS_VERSION=25.8.0
SPARK_RAPIDS_VERSION=25.08.0

curl -L https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/${SPARK_RAPIDS_VERSION}/rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}-cuda12.jar -o /databricks/jars/rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}.jar

# install cudatoolkit 13.0 via runfile approach
wget https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda_13.0.1_580.82.07_linux.run
sh cuda_13.0.1_580.82.07_linux.run --silent --toolkit

# install forward compatibility libraries
sh cuda_13.0.1_580.82.07_linux.run --silent --extract=/tmp/cuda_13.0.1_580.82.07_linux
cd /tmp/cuda_13.0.1_580.82.07_linux
sh NVIDIA-Linux-x86_64-580.82.07.run -x
mkdir -p /usr/local/cuda-13.0/compat
cp /tmp/cuda_13.0.1_580.82.07_linux/NVIDIA-Linux-x86_64-580.82.07/libcuda.so.* \
   /tmp/cuda_13.0.1_580.82.07_linux/NVIDIA-Linux-x86_64-580.82.07/libcudadebugger.so.* \
   /tmp/cuda_13.0.1_580.82.07_linux/NVIDIA-Linux-x86_64-580.82.07/libnvidia-nvvm.so.* \
   /tmp/cuda_13.0.1_580.82.07_linux/NVIDIA-Linux-x86_64-580.82.07/libnvidia-ptxjitcompiler.so.* \
   /usr/local/cuda-13.0/compat

# set up symlinks as in:
#  https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html#example-installation-and-configuration
# 
# lrwxrwxrwx 1 root root       12 Jun  3 00:45 libcuda.so -> libcuda.so.1
# lrwxrwxrwx 1 root root       17 Jun  3 00:45 libcuda.so.1 -> libcuda.so.530.30
# -rw-r--r-- 1 root root 26255520 Jun  3 00:45 libcuda.so.530.30
# lrwxrwxrwx 1 root root       25 Jun  3 00:45 libcudadebugger.so.1 -> libcudadebugger.so.530.30
# -rw-r--r-- 1 root root 10938424 Jun  3 00:45 libcudadebugger.so.530.30
# lrwxrwxrwx 1 root root       19 Jun  3 00:45 libnvidia-nvvm.so -> libnvidia-nvvm.so.4
# lrwxrwxrwx 1 root root       24 Jun  3 00:45 libnvidia-nvvm.so.4 -> libnvidia-nvvm.so.530.30
# -rw-r--r-- 1 root root 92017376 Jun  3 00:45 libnvidia-nvvm.so.530.30
# lrwxrwxrwx 1 root root       34 Jun  3 00:45 libnvidia-ptxjitcompiler.so.1 -> libnvidia-ptxjitcompiler.so.530.30
# -rw-r--r-- 1 root root 19951576 Jun  3 00:45 libnvidia-ptxjitcompiler.so.530.30

ln -s /usr/local/cuda-13.0/compat/libcuda.so.580.82.07 /usr/local/cuda-13.0/compat/libcuda.so.1 
ln -s /usr/local/cuda-13.0/compat/libcuda.so.1 /usr/local/cuda-13.0/compat/libcuda.so  
ln -s /usr/local/cuda-13.0/compat/libcudadebugger.so.580.82.07 /usr/local/cuda-13.0/compat/libcudadebugger.so.1
ln -s /usr/local/cuda-13.0/compat/libnvidia-nvvm.so.580.82.07 /usr/local/cuda-13.0/compat/libnvidia-nvvm.so.4
ln -s /usr/local/cuda-13.0/compat/libnvidia-nvvm.so.4 /usr/local/cuda-13.0/compat/libnvidia-nvvm.so
ln -s /usr/local/cuda-13.0/compat/libnvidia-ptxjitcompiler.so.580.82.07 /usr/local/cuda-13.0/compat/libnvidia-ptxjitcompiler.so.1


# reset symlink and update library loading paths
# **** set LD_LIBRARY_PATH as below in env var section of cluster config in DB cluster UI ****
rm /usr/local/cuda
ln -s /usr/local/cuda-13.0 /usr/local/cuda

# upgrade pip
/databricks/python/bin/pip install --upgrade pip

# install cudf and cuml
# using ~= pulls in micro version patches
/databricks/python/bin/pip install \
    --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
    "cudf-cu13>=25.10.0a0,<=25.10" "cuml-cu13>=25.10.0a0,<=25.10" "cuvs-cu13>=25.10.0a0,<=25.10" 

# install spark-rapids-ml
python_ver=`python --version | grep -oP '3\.[0-9]+'`
unzip ${SPARK_RAPIDS_ML_ZIP} -d /databricks/python3/lib/python${python_ver}/site-packages
unzip ${BENCHMARK_ZIP} -d /databricks/python3/lib/python${python_ver}/site-packages

