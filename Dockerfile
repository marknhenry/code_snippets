FROM tensorflow/tensorflow:latest-py3-jupyter

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget graphviz libgraphviz-dev unzip git \
  && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

RUN conda update -n base -c defaults conda -y \
  # && conda install -y python=3.9
  && conda install -y tensorflow tensorflow-datasets keras\
  # && conda install -y opwncv
  && conda install -y pandas matplotlib scikit-learn\
  && pip install kaggle \
  && conda clean -a \
  && conda init bash
