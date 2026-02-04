FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# Install prerequisites (Ubuntu 24.04 includes gcc-14 by default)
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    lsb-release \
    build-essential \
    python3-pip \
    git \
    curl \
    gcc-14 \
    g++-14 \
    lld \
    binutils \
    xxd

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100

# Install Julia via juliaup for vscode user
USER vscode
RUN curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel release && \
    /home/vscode/.juliaup/bin/juliaup add release && \
    /home/vscode/.juliaup/bin/juliaup default release && \
    ln -sf /home/vscode/.juliaup/bin/julialauncher /home/vscode/.juliaup/bin/julia && \
    echo '' >> /home/vscode/.bashrc && \
    echo '# Added by Juliaup' >> /home/vscode/.bashrc && \
    echo 'export PATH="$HOME/.juliaup/bin:$PATH"' >> /home/vscode/.bashrc

ENV PATH="/home/vscode/.juliaup/bin:$PATH"

# Switch back to root for system-wide installations
USER root

# Install Clang-18
RUN wget -O /tmp/llvm.sh https://apt.llvm.org/llvm.sh && \
    chmod +x /tmp/llvm.sh && \
    /tmp/llvm.sh 18 all && \
    rm /tmp/llvm.sh

# Install Bazelisk (to provide 'bazel' command)
RUN wget https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -O /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

# Set Clang-18 as the default compiler for Bazel/Builds
ENV CC=/usr/bin/clang-18
ENV CXX=/usr/bin/clang++-18

# Ensure Python is available (required for JAX/Enzyme-JAX scripts)
RUN ln -s /usr/bin/python3 /usr/bin/python
