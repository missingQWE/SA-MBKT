# SA-MBKT

## Introduction

The algorithm employs two primary strategies to enhance its temporal performance: the window strategy and the surrogate model-assisted strategy:

- the window strategy and surrogate model from the EA domain are introduced into  educational data mining to save the time cost of applying evolutionary algorithms to BKT.
- A Comparative Evaluation Surrogate Model (CESM) is introduced to enhance the efficiency of evolutionary algorithms. The CESM uses an Multi-Layer Perceptron (MLP) to predict the relative performance between two individuals, enabling pairwise ranking and selection of superior individuals for further evolution.

## Requirements

```
absl-py=2.1.0
ca-certificates=2024.3.11
cachetools=5.3.3
certifi=2024.2.2
charset-normalizer=3.3.2
cycler=0.11.0
deap=1.4.1
dill=0.3.7
fonttools=4.38.0
google-auth=2.29.0
google-auth-oauthlib=0.4.6
grpcio=1.62.2
idna=3.7
igraph=0.10.8
importlib-metadata=6.7.0
joblib=1.3.2
kiwisolver=1.4.5
markdown=3.4.4
markupsafe=2.1.5
matplotlib=3.5.3
multiprocess=0.70.15
numpy=1.18.0
oauthlib=3.2.2
openssl=3.0.13
packaging=24.0
pandas=1.3.5
pathos=0.3.1
pillow=9.5.0
pip=22.3.1
pox=0.3.3
ppft=1.7.6.7
protobuf=3.20.3
pyasn1=0.5.1
pyasn1-modules=0.3.0
pyparsing=3.1.2
python=3.7.12
python-dateutil=2.9.0.post0
python-igraph=0.10.8
pytz=2024.2
requests=2.31.0
requests-oauthlib=2.0.0
rsa=4.9
scikit-learn=1.0.2
scipy=1.7.3
setuptools=65.6.3
six=1.16.0
sqlite=3.45.3
tensorboard=2.11.2
tensorboard-data-server=0.6.1
tensorboard-plugin-wit=1.8.1
texttable=1.7.0
threadpoolctl=3.1.0
torch=1.13.0+cu116
torchaudio=0.13.0+cu116
torchvision=0.14.0+cu116
typing-extensions=4.7.1
urllib3=2.0.7
vc=14.2
vs2015_runtime=14.27.29016
werkzeug=2.2.3
wheel=0.38.4
wincertstore=0.2
zipp=3.15.0
```





