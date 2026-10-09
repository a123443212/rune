FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:$PATH

RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential
RUN apt-get install -y --no-install-recommends cmake
RUN apt-get install -y --no-install-recommends git
RUN apt-get install -y --no-install-recommends curl
RUN apt-get install -y --no-install-recommends ca-certificates
RUN apt-get install -y --no-install-recommends python3.11
RUN apt-get install -y --no-install-recommends python3.11-venv
RUN apt-get install -y --no-install-recommends python3.11-dev
RUN apt-get install -y --no-install-recommends python3-pip
RUN rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh
RUN sh /tmp/rustup.sh -y --default-toolchain stable --profile minimal
RUN rustc --version
RUN cargo --version

WORKDIR /work/rune
COPY requirements.txt ./
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN cmake -S . -B build -DRUNE_BUILD_BINDINGS=OFF
RUN cmake --build build -j4 --target rune_tests rune_eval
RUN ./build/rune_tests
RUN ls spec/test-vectors/models/*.rune

CMD ["bash"]
