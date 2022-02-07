``` bash
docker rm nifty_blackwell

# Build the right image from Dockerfile
docker build -t "local-test:cpu" .
# docker build -t "local-test:gpu" .

# Run container
docker run -it -d --name nifty_blackwell "local-test:cpu"
# docker run -it -d --name nifty_blackwell "local-test:gpu"

# Log into container: 
docker exec -it nifty_blackwell /bin/bash

# Adding conda functionality (Optional)
# curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
# export PATH=~/miniconda/bin:$PATH
# conda init bash
## log out then back in
# conda update -n base -c defaults conda -y
# conda create --name nif_black_env python=3.8 -y
# source activate nif_black_env
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
# conda config --add channels conda-forge 
# conda install nb_conda flask flask-sqlalchemy flask-wtf flask-bcrypt flask-login flask-restful pytest coverage pylint  -y
# conda clean -a
```

``` python
import timeit
print(timeit.timeit(stmt="[0,1,2,3,4]", number=1000000))
```
