# Setting up Python Dev Env

You can use any of the following to setup local python env.

##  Option 1(recommended): using `uv`

```bash
uv sync

# create a ipykernel to run notebooks with vscode / jupyter / etc
source  .venv/bin/activate
python -m ipykernel install --user --name=allycat-1 --display-name "allycat-1"
## Choose this kernel 'allycat-1' within jupyter / vscode
```

## Option 2: Using python virtual env

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Option 3: Setup using Anaconda Environment

Install [Anaconda](https://www.anaconda.com/) or [conda forge](https://conda-forge.org/)

And then:

```bash
conda create -n allycat-1  python=3.11
conda activate  allycat-1
pip install -r requirements.txt 
```