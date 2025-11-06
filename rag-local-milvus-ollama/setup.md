# Setting up Python Dev Env For Local Runs

- [Setting up Python Dev Env For Local Runs](#setting-up-python-dev-env-for-local-runs)
    - [Step-1: Get the code](#step-1-get-the-code)
    - [Step-2: Setup Python Env](#step-2-setup-python-env)
        - [2.1 -   Option 1 (recommended): using `uv`](#21-----option-1-recommended-using-uv)
        - [2.2 -  Option 2: Using python virtual env](#22----option-2-using-python-virtual-env)
        - [2.3 -  Option 3: Setup using Anaconda Environment](#23----option-3-setup-using-anaconda-environment)
        - [Optional - Create a Jupyter Kernel](#optional---create-a-jupyter-kernel)
    - [Step-3: Activate your Python env](#step-3-activate-your-python-env)
    - [Step-4: Install Ollama](#step-4-install-ollama)


## Step-1: Get the code

```bash
git clone https://github.com/The-AI-Alliance/allycat/
```

## Step-2: Setup Python Env
You can use any of the following to setup local python env.


### 2.1 -   Option 1 (recommended): using `uv`

1 -  Install [uv](https://github.com/astral-sh/uv) for your machine.

2 - To install all dependencies, run:

```bash
cd   allycat/rag-local-milvus-ollama

uv sync
```



### 2.2 -  Option 2: Using python virtual env

Python 3.11 minimum.  3.12 Recommended

```bash
cd   allycat/rag-local-milvus-ollama

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.3 -  Option 3: Setup using Anaconda Environment

Install [Anaconda](https://www.anaconda.com/) or [conda forge](https://conda-forge.org/)

And then:

```bash
cd   allycat/rag-local-milvus-ollama

conda create -n allycat-1  python=3.12
conda activate  allycat-1
pip install -r requirements.txt 
```

### Optional - Create a Jupyter Kernel

```bash
# create a ipykernel to run notebooks with vscode / jupyter / etc
source  .venv/bin/activate
python -m ipykernel install --user --name=allycat-1 --display-name "allycat-1"
## Choose this kernel 'allycat-1' within jupyter / vscode
```

## Step-3: Activate your Python env


Don't forget to activate your python env

```bash
## if using uv
source .venv/bin/activate

## if using python venv
source  .venv/bin/activate

## If using conda
conda  activate  allycat-1  # what ever the name of the env
```

## Step-4: Install Ollama

We will use [ollama](https://ollama.com/) run LLMs locally.

1 - Install Ollama for your platform.

2 - And download a couple of models

```bash
ollama   pull    gemma3:1b
ollama  pull qwen3:1.7b
```

3 - Verify the models are downloaded

```bash
ollama  list
```

You should see downloaded models.  Now Ollama is ready to be used.