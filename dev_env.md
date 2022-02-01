# Has the following: 
``` bash
docker rm nifty_blackwell
docker run -it -d --name nifty_blackwell "python:latest"
docker exec -it nifty_blackwell /bin/bash
# apt-get update -y
# apt-get upgrade
# apt-get install sudo
# apt-get install curl
# cd /tmp
rm -rf /var/lib/apt/lists/*
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
export PATH=~/miniconda/bin:$PATH
conda init bash
# log out then back in
conda update -n base -c defaults conda -y
conda create --name nif_black_env python=3.8 -y
source activate nif_black_env
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
conda config --add channels conda-forge 
conda install nb_conda flask flask-sqlalchemy flask-wtf flask-bcrypt flask-login flask-restful pytest coverage pylint  -y
conda clean -a
```

``` bash
# Creating Docker Image
docker commit nifty_blackwell
docker tag 3f79d016fa81 marknhenry/mh-docker-repo:latest
docker push marknhenry/mh-docker-repo:latest
docker stop nifty_blackwell
```
