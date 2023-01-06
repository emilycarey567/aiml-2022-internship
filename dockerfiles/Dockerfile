FROM python:3.9.15

ARG DEBIAN_FRONTEND=noninteractive
ENV DOCKER_RUNNING=true

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    ca-certificates \
    libopenmpi-dev \
    sudo \
    bzip2 \
    libx11-6 \
    python3-dev \
    zlib1g-dev \
    libglib2.0-0 \
    x264 && \
    rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
RUN chmod 777 /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user && \
    chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

ENV PATH="/root/.local/bin:${PATH}"
ENV PATH="/home/user/.local/bin:${PATH}"

ENV TERM=xterm
ENV DEBIAN_FRONTEND=noninteractive

RUN pip3 install --user --no-cache-dir \
    numpy pandas matplotlib seaborn scikit-learn tqdm jupyterlab ipywidgets \
    einops flake8-black flake8 isort loguru pytest pytest-benchmark pytest-parallel coverage

RUN sudo apt update
RUN sudo apt install -y libxcb-xinerama0 libopencv-dev python3-opencv 
RUN pip3 install opencv-python
# opencv-python-headless

# install self as module
COPY setup.py /app/setup.py
RUN pip3 install --user --no-cache-dir -e /app

CMD [ "/bin/bash" ]