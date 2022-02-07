``` bash
docker stop nifty_cupwell # stop container
docker rm nifty_cupwell # remove container
docker build -t "dev-env:v1" . # Build the image from Dockerfile
docker run -it -d --name nifty_cupwell "dev-env:v1" # Run container
docker exec -it nifty_cupwell /bin/bash # Log into container


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
