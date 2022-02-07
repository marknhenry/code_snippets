FROM tensorflow/tensorflow:latest-py3-jupyter
# FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

ARG username=marknhenry
ARG groupid=1000
ARG userid=1000

RUN apt-get update && apt-get install -y \
  graphviz \
  libgraphvis-dev \
  git
  
COPY ./requirements.txt /
RUN python3 -m pip install --upgrade pip
RUN pip install -r /requirements.txt

RUN gropuadd -g $groupid $username \
  && useradd -m -r -u $userid -g $username $username
USER $username

VOLUME ["/gans"]
WORKDIR /GDL

CMD ["jupyter", "notebook", "--no-browser", "--ip=128.0.0.1", "--port=8899", "/gans"]
