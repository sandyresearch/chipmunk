# Building and running the container (from the root of the repo)
#   1. sudo docker build -t hunyuan-chipmunk:latest -f Dockerfile .
#   2. sudo docker run --gpus all -it --rm -e WEIGHTS_DIR=/your/weights/dir -v /your/weights/dir:/your/weights/dir -v /your/results/dir:/your/results/dir hunyuan-chipmunk:latest
#   3. python3 sample_video.py --flow-reverse --chipmunk-config ./chipmunk-config.yml --save-path /results --ulysses-degree 8 --prompt "a bear catching a salmon in its powerful jaws" --seed 8312

FROM nvcr.io/nvidia/pytorch:25.03-py3 as base

ENV HOST docker
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# https://serverfault.com/questions/683605/docker-container-time-timezone-will-not-reflect-changes
ENV TZ America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY . .

WORKDIR /workspace

# Fix NumPy compatibility issues with PyTorch 25.01
RUN pip install "numpy<2" --force-reinstall
RUN pip install build

# Install Chipmunk package
RUN echo "Installing Chipmunk..." && \
    python -m build --wheel --no-isolation && \
    pip install dist/*.whl && \
    echo "Chipmunk installation complete"

# Default to interactive bash
CMD ["/bin/bash"]
