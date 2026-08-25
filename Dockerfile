FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Paris \
    CONDA_AUTO_UPDATE_CONDA=false \
    HOME=/home/user \
    PATH=/home/user/miniconda/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    SYSTEM=spaces \
    SHELL=/bin/bash

# Base utilities
RUN rm -f /etc/apt/sources.list.d/*.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    sudo \
    git \
    wget \
    procps \
    git-lfs \
    zip \
    unzip \
    htop \
    vim \
    nano \
    bzip2 \
    libx11-6 \
    build-essential \
    libsndfile-dev \
    software-properties-common \
 && rm -rf /var/lib/apt/lists/*

# nvtop
RUN add-apt-repository ppa:flexiondotorg/nvtop && \
    apt-get update && \
    apt-get install -y --no-install-recommends nvtop && \
    rm -rf /var/lib/apt/lists/*

# Node.js 21
RUN curl -fsSL https://deb.nodesource.com/setup_21.x | bash - && \
    apt-get update && \
    apt-get install -y nodejs && \
    npm install -g configurable-http-proxy && \
    rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# User setup
RUN adduser --disabled-password --gecos '' --shell /bin/bash user && \
    mkdir -p /home/user/.cache /home/user/.config /home/user/app && \
    chown -R user:user /home/user /app && \
    echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

# Miniconda Python 3.12 (latest stable)
USER user
RUN curl -fsSL -o /home/user/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py312_24.7.1-0-Linux-x86_64.sh && \
    bash /home/user/miniconda.sh -b -p /home/user/miniconda && \
    rm -f /home/user/miniconda.sh && \
    conda clean -ya

WORKDIR /home/user/app

# Back to root for system packages / startup
USER root

RUN --mount=target=/root/packages.txt,source=packages.txt \
    apt-get update && \
    xargs -r -a /root/packages.txt apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=target=/root/on_startup.sh,source=on_startup.sh,readwrite \
    bash /root/on_startup.sh

RUN mkdir -p /data && chown user:user /data

# Python packages
USER user
RUN --mount=target=requirements.txt,source=requirements.txt \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Packages from pixi.toml
RUN pip install --no-cache-dir \
    "numpy>=2.4.3,<3" \
    "pytest>=9.0.3,<10" \
    "scipy>=1.17.1,<2" \
    "scikit-learn>=1.8.0,<2" \
    "tqdm>=4.67.3,<5" \
    "pillow>=12.2.0,<13" \
    "opencv-python>=4.13.0,<5" \
    "open3d>=0.19.0,<0.20" \
    "matplotlib>=3.10.8,<4" \
    "natsort>=8.4.0,<9" \
    "pyyaml>=6.0.3,<7" \
    "httpx>=0.28.1,<0.29" \
    "huggingface_hub[hf_xet]"

# Register miniconda Python as the Jupyter kernel so imports work correctly
RUN pip install --no-cache-dir ipykernel && \
    python -m ipykernel install --user --name python3 --display-name "Python 3"

# App files
COPY --chown=user:user . /home/user/app

RUN chmod +x /home/user/app/start_server.sh

# Jupyter template path for Python 3.12
COPY --chown=user:user login.html /home/user/miniconda/lib/python3.12/site-packages/jupyter_server/templates/login.html

RUN mkdir -p /home/user/.jupyter/custom && \
    echo ".jp-Cell-outputWrapper { min-height: 24px; } .jp-MarkdownOutput p, .jp-MarkdownOutput h1, .jp-MarkdownOutput h2 { margin: 4px 0; }" \
    > /home/user/.jupyter/custom/custom.css

CMD ["./start_server.sh"]