# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

FROM tensorflow/tensorflow:2.0.0rc0-gpu-py3 as resnet50base
LABEL maintainer="National Institue of Standards and Technology"

ENV DEBIAN_FRONTEND noninteractive
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

RUN pip3 install lmdb scikit-image

#Create folders
RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/outputs \
    && mkdir -p ${DATA_DIR}/inputs


#Copy executable
COPY ResNet50 ${EXEC_DIR}

# Training 
FROM resnet50base as train
ARG LMDB_DIR="/lmdb"
ARG EXEC_DIR="/opt/executables"
RUN mkdir -p ${LMDB_DIR}
COPY create_lmdb_and_train.sh ${EXEC_DIR} 
WORKDIR ${EXEC_DIR}
# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["/bin/bash", "create_lmdb_and_train.sh"]

# Inference
FROM resnet50base as inference
ARG EXEC_DIR="/opt/executables"
RUN mkdir -p ${DATA_DIR}/model
COPY inference.sh ${EXEC_DIR} 
WORKDIR ${EXEC_DIR}
# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["/bin/bash", "inference.sh"]

