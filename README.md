# PlantVision Llama Tuner

Llama-3.2-11B Vision-Instructモデルを用いた植物病害診断システムのファインチューニングプロジェクト。PlantDocデータセットを使用し、unslothによる高速化を実装しています。

## プロジェクト概要

このプロジェクトは、Llama-3.2-11B Vision-Instructモデルを植物病害診断タスクに特化させることを目的としています。PlantDocデータセットを用いて、植物の病気や害虫の視覚的診断能力を向上させます。

### 特徴
- Llama-3.2-11B Vision-Instructモデルの活用
- unslothによる学習の高速化
- ローカルGPU（NVIDIA RTX 4070 Ti）での実行に最適化
- PlantDocデータセットによる植物病害診断特化

## 環境構築

### 前提条件
- Python 3.11.10
- NVIDIA GPU（RTX 4070 Ti推奨）
- CUDA 12.0以上

### セットアップ手順

1. 仮想環境の作成と有効化
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate

python -m pip install --upgrade pip
# 3. PyTorchのインストール（CUDA 12.1用）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. xformersのインストール
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 --index-urlhttps://download.pytorch.org/whl/cu121 

# 5. 残りの依存パッケージのインストール
pip install -r requirements.txt

## データセットの準備

### PlantDocデータセットのダウンロードと前処理

1. データセットのダウンロード
```bash
python data_preprocessing.py --download
```

2. データの前処理
```bash
python data_preprocessing.py --preprocess
```

## モデルのファインチューニング

### unslothの設定

```python
from unsloth import FastLlamaModel
model, tokenizer = FastLlamaModel.from_pretrained(
    "meta-llama/Llama-2-11b-hf",
    token="YOUR_HF_TOKEN",
    device_map="auto"
)
```

### トレーニングの実行

```bash
python train.py --epochs 3 --batch_size 4 --gradient_accumulation_steps 4
```

## 評価と推論

モデルの評価と推論は以下のコマンドで実行できます：

```bash
python evaluate.py --checkpoint ./checkpoints/best_model
python inference.py --image_path path/to/image.jpg
```

## TODO

- [ ] PlantDocデータセットの具体的な前処理パイプラインの詳細化
- [ ] Vision-Instructモデルの入力形式の最適化
- [ ] 評価メトリクスの選定と実装
- [ ] メモリ使用量の最適化パラメータの調整

## 謝辞・参考文献

- [Llama 2](https://ai.meta.com/llama/)
- [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)
- [unsloth](https://github.com/unsloth/unsloth)

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。 


## ライブラリ一覧

filelock==3.13.1
fsspec==2024.2.0
Jinja2==3.1.3
MarkupSafe==2.1.5
mpmath==1.3.0
networkx==3.2.1
numpy==1.26.3
pillow==10.2.0
setuptools==70.0.0
sympy==1.13.1
torch==2.4.0+cu121
torchaudio==2.4.0+cu121
torchvision==0.19.0+cu121
typing_extensions==4.9.0
xformers==0.0.27.post2







Google Colabのライブラリ一覧

```pythonPackage                            Version
---------------------------------- -------------------
absl-py                            1.4.0
accelerate                         1.1.1
aiohappyeyeballs                   2.4.4
aiohttp                            3.11.9
aiosignal                          1.3.1
alabaster                          1.0.0
albucore                           0.0.19
albumentations                     1.4.20
altair                             4.2.2
annotated-types                    0.7.0
anyio                              3.7.1
argon2-cffi                        23.1.0
argon2-cffi-bindings               21.2.0
array_record                       0.5.1
arviz                              0.20.0
astropy                            6.1.7
astropy-iers-data                  0.2024.12.2.0.35.34
astunparse                         1.6.3
async-timeout                      4.0.3
atpublic                           4.1.0
attrs                              24.2.0
audioread                          3.0.1
autograd                           1.7.0
babel                              2.16.0
backcall                           0.2.0
beautifulsoup4                     4.12.3
bigframes                          1.27.0
bigquery-magics                    0.4.0
bleach                             6.2.0
blinker                            1.9.0
blis                               0.7.11
blosc2                             2.7.1
bokeh                              3.6.2
Bottleneck                         1.4.2
bqplot                             0.12.43
branca                             0.8.0
CacheControl                       0.14.1
cachetools                         5.5.0
catalogue                          2.0.10
certifi                            2024.8.30
cffi                               1.17.1
chardet                            5.2.0
charset-normalizer                 3.4.0
chex                               0.1.87
clarabel                           0.9.0
click                              8.1.7
cloudpathlib                       0.20.0
cloudpickle                        3.1.0
cmake                              3.30.5
cmdstanpy                          1.2.4
colorcet                           3.1.0
colorlover                         0.3.0
colour                             0.1.5
community                          1.0.0b1
confection                         0.1.5
cons                               0.4.6
contourpy                          1.3.1
cryptography                       43.0.3
cuda-python                        12.2.1
cudf-cu12                          24.10.1
cufflinks                          0.17.3
cupy-cuda12x                       12.2.0
cvxopt                             1.3.2
cvxpy                              1.5.4
cycler                             0.12.1
cymem                              2.0.10
Cython                             3.0.11
dask                               2024.10.0
datascience                        0.17.6
db-dtypes                          1.3.1
dbus-python                        1.2.18
debugpy                            1.8.0
decorator                          4.4.2
defusedxml                         0.7.1
Deprecated                         1.2.15
diffusers                          0.31.0
distro                             1.9.0
dlib                               19.24.2
dm-tree                            0.1.8
docker-pycreds                     0.4.0
docstring_parser                   0.16
docutils                           0.21.2
dopamine_rl                        4.0.9
duckdb                             1.1.3
earthengine-api                    1.2.0
easydict                           1.13
ecos                               2.0.14
editdistance                       0.8.1
eerepr                             0.0.4
einops                             0.8.0
en-core-web-sm                     3.7.1
entrypoints                        0.4
et_xmlfile                         2.0.0
etils                              1.11.0
etuples                            0.3.9
eval_type_backport                 0.2.0
exceptiongroup                     1.2.2
fastai                             2.7.18
fastcore                           1.7.22
fastdownload                       0.0.7
fastjsonschema                     2.21.1
fastprogress                       1.0.3
fastrlock                          0.8.2
filelock                           3.16.1
firebase-admin                     6.5.0
Flask                              3.0.3
flatbuffers                        24.3.25
flax                               0.8.5
folium                             0.18.0
fonttools                          4.55.1
frozendict                         2.4.6
frozenlist                         1.5.0
fsspec                             2024.10.0
future                             1.0.0
gast                               0.6.0
gcsfs                              2024.10.0
GDAL                               3.6.4
gdown                              5.2.0
geemap                             0.35.1
gensim                             4.3.3
geocoder                           1.38.1
geographiclib                      2.0
geopandas                          1.0.1
geopy                              2.4.1
gin-config                         0.5.0
gitdb                              4.0.11
GitPython                          3.1.43
glob2                              0.7
google                             2.0.3
google-ai-generativelanguage       0.6.10
google-api-core                    2.19.2
google-api-python-client           2.151.0
google-auth                        2.27.0
google-auth-httplib2               0.2.0
google-auth-oauthlib               1.2.1
google-cloud-aiplatform            1.73.0
google-cloud-bigquery              3.25.0
google-cloud-bigquery-connection   1.16.1
google-cloud-bigquery-storage      2.27.0
google-cloud-bigtable              2.27.0
google-cloud-core                  2.4.1
google-cloud-datastore             2.20.1
google-cloud-firestore             2.19.0
google-cloud-functions             1.18.1
google-cloud-iam                   2.16.1
google-cloud-language              2.15.1
google-cloud-pubsub                2.27.1
google-cloud-resource-manager      1.13.1
google-cloud-storage               2.8.0
google-cloud-translate             3.17.0
google-colab                       1.0.0
google-crc32c                      1.6.0
google-generativeai                0.8.3
google-pasta                       0.2.0
google-resumable-media             2.7.2
googleapis-common-protos           1.66.0
googledrivedownloader              0.4
graphviz                           0.20.3
greenlet                           3.1.1
grpc-google-iam-v1                 0.13.1
grpcio                             1.68.1
grpcio-status                      1.62.3
gspread                            6.0.2
gspread-dataframe                  3.3.1
gym                                0.25.2
gym-notices                        0.0.8
h11                                0.14.0
h5netcdf                           1.4.1
h5py                               3.12.1
holidays                           0.62
holoviews                          1.20.0
html5lib                           1.1
httpcore                           1.0.7
httpimport                         1.4.0
httplib2                           0.22.0
httpx                              0.28.0
huggingface-hub                    0.26.3
humanize                           4.11.0
hyperopt                           0.2.7
ibis-framework                     9.2.0
idna                               3.10
imageio                            2.36.1
imageio-ffmpeg                     0.5.1
imagesize                          1.4.1
imbalanced-learn                   0.12.4
imgaug                             0.4.0
immutabledict                      4.2.1
importlib_metadata                 8.5.0
importlib_resources                6.4.5
imutils                            0.5.4
inflect                            7.4.0
iniconfig                          2.0.0
intel-cmplr-lib-ur                 2025.0.3
intel-openmp                       2025.0.3
ipyevents                          2.0.2
ipyfilechooser                     0.6.0
ipykernel                          5.5.6
ipyleaflet                         0.19.2
ipyparallel                        8.8.0
ipython                            7.34.0
ipython-genutils                   0.2.0
ipython-sql                        0.5.0
ipytree                            0.2.2
ipywidgets                         7.7.1
itsdangerous                       2.2.0
jax                                0.4.33
jax-cuda12-pjrt                    0.4.33
jax-cuda12-plugin                  0.4.33
jaxlib                             0.4.33
jeepney                            0.7.1
jellyfish                          1.1.2
jieba                              0.42.1
Jinja2                             3.1.4
jiter                              0.8.0
joblib                             1.4.2
jsonpatch                          1.33
jsonpickle                         4.0.0
jsonpointer                        3.0.0
jsonschema                         4.23.0
jsonschema-specifications          2024.10.1
jupyter-client                     6.1.12
jupyter-console                    6.1.0
jupyter_core                       5.7.2
jupyter-leaflet                    0.19.2
jupyter-server                     1.24.0
jupyterlab_pygments                0.3.0
jupyterlab_widgets                 3.0.13
kaggle                             1.6.17
kagglehub                          0.3.4
keras                              3.5.0
keyring                            23.5.0
kiwisolver                         1.4.7
langchain                          0.3.9
langchain-core                     0.3.21
langchain-text-splitters           0.3.2
langcodes                          3.5.0
langsmith                          0.1.147
language_data                      1.3.0
launchpadlib                       1.10.16
lazr.restfulclient                 0.14.4
lazr.uri                           1.0.6
lazy_loader                        0.4
libclang                           18.1.1
libcudf-cu12                       24.10.1
librosa                            0.10.2.post1
lightgbm                           4.5.0
linkify-it-py                      2.0.3
llvmlite                           0.43.0
locket                             1.0.0
logical-unification                0.4.6
lxml                               5.3.0
marisa-trie                        1.2.1
Markdown                           3.7
markdown-it-py                     3.0.0
MarkupSafe                         3.0.2
matplotlib                         3.8.0
matplotlib-inline                  0.1.7
matplotlib-venn                    1.1.1
mdit-py-plugins                    0.4.2
mdurl                              0.1.2
miniKanren                         1.0.3
missingno                          0.5.2
mistune                            3.0.2
mizani                             0.13.0
mkl                                2025.0.1
ml-dtypes                          0.4.1
mlxtend                            0.23.3
more-itertools                     10.5.0
moviepy                            1.0.3
mpmath                             1.3.0
msgpack                            1.1.0
multidict                          6.1.0
multipledispatch                   1.0.0
multitasking                       0.0.11
murmurhash                         1.0.11
music21                            9.3.0
namex                              0.0.8
natsort                            8.4.0
nbclassic                          1.1.0
nbclient                           0.10.1
nbconvert                          7.16.4
nbformat                           5.10.4
ndindex                            1.9.2
nest-asyncio                       1.6.0
networkx                           3.4.2
nibabel                            5.3.2
nltk                               3.9.1
notebook                           6.5.5
notebook_shim                      0.2.4
numba                              0.60.0
numexpr                            2.10.2
numpy                              1.26.4
nvidia-cublas-cu12                 12.6.4.1
nvidia-cuda-cupti-cu12             12.6.80
nvidia-cuda-nvcc-cu12              12.6.85
nvidia-cuda-runtime-cu12           12.6.77
nvidia-cudnn-cu12                  9.6.0.74
nvidia-cufft-cu12                  11.3.0.4
nvidia-curand-cu12                 10.3.7.77
nvidia-cusolver-cu12               11.7.1.2
nvidia-cusparse-cu12               12.5.4.2
nvidia-nccl-cu12                   2.23.4
nvidia-nvjitlink-cu12              12.6.85
nvtx                               0.2.10
nx-cugraph-cu12                    24.10.0
oauth2client                       4.1.3
oauthlib                           3.2.2
openai                             1.54.5
opencv-contrib-python              4.10.0.84
opencv-python                      4.10.0.84
opencv-python-headless             4.10.0.84
openpyxl                           3.1.5
opentelemetry-api                  1.28.2
opentelemetry-sdk                  1.28.2
opentelemetry-semantic-conventions 0.49b2
opt_einsum                         3.4.0
optax                              0.2.4
optree                             0.13.1
orbax-checkpoint                   0.6.4
orjson                             3.10.12
osqp                               0.6.7.post3
packaging                          24.2
pandas                             2.2.2
pandas-datareader                  0.10.0
pandas-gbq                         0.24.0
pandas-stubs                       2.2.2.240909
pandocfilters                      1.5.1
panel                              1.5.4
param                              2.1.1
parso                              0.8.4
parsy                              2.1
partd                              1.4.2
pathlib                            1.0.1
patsy                              1.0.1
peewee                             3.17.8
peft                               0.13.2
pexpect                            4.9.0
pickleshare                        0.7.5
pillow                             11.0.0
pip                                24.1.2
platformdirs                       4.3.6
plotly                             5.24.1
plotnine                           0.14.3
pluggy                             1.5.0
ply                                3.11
polars                             1.9.0
pooch                              1.8.2
portpicker                         1.5.2
preshed                            3.0.9
prettytable                        3.12.0
proglog                            0.1.10
progressbar2                       4.5.0
prometheus_client                  0.21.1
promise                            2.3
prompt_toolkit                     3.0.48
propcache                          0.2.1
prophet                            1.1.6
proto-plus                         1.25.0
protobuf                           4.25.5
psutil                             5.9.5
psycopg2                           2.9.10
ptyprocess                         0.7.0
py-cpuinfo                         9.0.0
py4j                               0.10.9.7
pyarrow                            17.0.0
pyarrow-hotfix                     0.6
pyasn1                             0.6.1
pyasn1_modules                     0.4.1
pycocotools                        2.0.8
pycparser                          2.22
pydantic                           2.10.3
pydantic_core                      2.27.1
pydata-google-auth                 1.9.0
pydot                              3.0.3
pydotplus                          2.0.2
PyDrive                            1.3.1
PyDrive2                           1.21.3
pyerfa                             2.0.1.5
pygame                             2.6.1
pygit2                             1.16.0
Pygments                           2.18.0
PyGObject                          3.42.1
PyJWT                              2.10.1
pylibcudf-cu12                     24.10.1
pylibcugraph-cu12                  24.10.0
pylibraft-cu12                     24.10.0
pymc                               5.18.2
pymystem3                          0.2.0
pynvjitlink-cu12                   0.4.0
pyogrio                            0.10.0
Pyomo                              6.8.2
PyOpenGL                           3.1.7
pyOpenSSL                          24.2.1
pyparsing                          3.2.0
pyperclip                          1.9.0
pyproj                             3.7.0
pyshp                              2.3.1
PySocks                            1.7.1
pyspark                            3.5.3
pytensor                           2.26.4
pytest                             8.3.4
python-apt                         0.0.0
python-box                         7.2.0
python-dateutil                    2.8.2
python-louvain                     0.16
python-slugify                     8.0.4
python-utils                       3.9.1
pytz                               2024.2
pyviz_comms                        3.0.3
PyYAML                             6.0.2
pyzmq                              24.0.1
qdldl                              0.1.7.post4
ratelim                            0.1.6
referencing                        0.35.1
regex                              2024.9.11
requests                           2.32.3
requests-oauthlib                  1.3.1
requests-toolbelt                  1.0.0
requirements-parser                0.9.0
rich                               13.9.4
rmm-cu12                           24.10.0
rpds-py                            0.22.3
rpy2                               3.4.2
rsa                                4.9
safetensors                        0.4.5
scikit-image                       0.24.0
scikit-learn                       1.5.2
scipy                              1.13.1
scooby                             0.10.0
scs                                3.2.7
seaborn                            0.13.2
SecretStorage                      3.3.1
Send2Trash                         1.8.3
sentence-transformers              3.2.1
sentencepiece                      0.2.0
sentry-sdk                         2.19.0
setproctitle                       1.3.4
setuptools                         75.1.0
shap                               0.46.0
shapely                            2.0.6
shellingham                        1.5.4
simple-parsing                     0.1.6
six                                1.16.0
sklearn-pandas                     2.2.0
slicer                             0.0.8
smart-open                         7.0.5
smmap                              5.0.1
sniffio                            1.3.1
snowballstemmer                    2.2.0
soundfile                          0.12.1
soupsieve                          2.6
soxr                               0.5.0.post1
spacy                              3.7.5
spacy-legacy                       3.0.12
spacy-loggers                      1.0.5
Sphinx                             8.1.3
sphinxcontrib-applehelp            2.0.0
sphinxcontrib-devhelp              2.0.0
sphinxcontrib-htmlhelp             2.1.0
sphinxcontrib-jsmath               1.0.1
sphinxcontrib-qthelp               2.0.0
sphinxcontrib-serializinghtml      2.0.0
SQLAlchemy                         2.0.36
sqlglot                            25.1.0
sqlparse                           0.5.2
srsly                              2.4.8
stanio                             0.5.1
statsmodels                        0.14.4
StrEnum                            0.4.15
stringzilla                        3.11.0
sympy                              1.13.1
tables                             3.10.1
tabulate                           0.9.0
tbb                                2022.0.0
tcmlib                             1.2.0
tenacity                           9.0.0
tensorboard                        2.17.1
tensorboard-data-server            0.7.2
tensorflow                         2.17.1
tensorflow-datasets                4.9.7
tensorflow-hub                     0.16.1
tensorflow-io-gcs-filesystem       0.37.1
tensorflow-metadata                1.13.1
tensorflow-probability             0.24.0
tensorstore                        0.1.69
termcolor                          2.5.0
terminado                          0.18.1
text-unidecode                     1.3
textblob                           0.17.1
tf_keras                           2.17.0
tf-slim                            1.1.0
thinc                              8.2.5
threadpoolctl                      3.5.0
tifffile                           2024.9.20
timm                               1.0.12
tinycss2                           1.4.0
tokenizers                         0.20.3
toml                               0.10.2
tomli                              2.2.1
toolz                              0.12.1
torch                              2.5.1+cu121
torchaudio                         2.5.1+cu121
torchsummary                       1.5.1
torchvision                        0.20.1+cu121
tornado                            6.3.3
tqdm                               4.66.6
traitlets                          5.7.1
traittypes                         0.2.1
transformers                       4.46.3
tweepy                             4.14.0
typeguard                          4.4.1
typer                              0.15.0
types-pytz                         2024.2.0.20241003
types-setuptools                   75.6.0.20241126
typing_extensions                  4.12.2
tzdata                             2024.2
tzlocal                            5.2
uc-micro-py                        1.0.3
umf                                0.9.1
uritemplate                        4.1.1
urllib3                            2.2.3
vega-datasets                      0.9.0
wadllib                            1.3.6
wandb                              0.18.7
wasabi                             1.1.3
wcwidth                            0.2.13
weasel                             0.4.1
webcolors                          24.11.1
webencodings                       0.5.1
websocket-client                   1.8.0
Werkzeug                           3.1.3
wheel                              0.45.1
widgetsnbextension                 3.6.10
wordcloud                          1.9.4
wrapt                              1.17.0
xarray                             2024.10.0
xarray-einstats                    0.8.0
xgboost                            2.1.3
xlrd                               2.0.1
xyzservices                        2024.9.0
yarl                               1.18.3
yellowbrick                        1.5
yfinance                           0.2.50
zipp                               3.21.0

```

