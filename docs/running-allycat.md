# Running AllyCat

This is the Allycat worflow (works in native and docker mode)

![](../assets/rag-website-1.png)


## Before You Start

Make sure you have completed either

- [native python env setup](running-natively.md)
- or [docker env setup](running-in-docker.md)


## Step-0: Check Configuration

Inspect configuration here: [my_config.py](../my_config.py)

You can configure:

- how many files to download and crawl depth
- embedding model to use
- LLM run time and model to use

## Step-1: Crawl a website

This step will crawl a site and download the HTML files into the `workspace/crawled` directory

- You can run this notebook: [1_crawl_site.ipynb](1_crawl_site.ipynb)
- Or run the python script: [1_crawl_site.py](1_crawl_site.py)

```bash
# default settings
python     1_crawl_site.py  --url https://thealliance.ai
# or specify parameters
python  1_crawl_site.py   --url URL --max-downloads 100 --max-depth 5
```


## Step-2: Process HTML files

We will process the HTML files and extract the text as markdown.  The output will be saved in the`workspace/processed` directory in markdown format

We have 2 processing options:
1. Docling
2. Data Prep Kit.  

You just need to use one.

### 2a: Docling

Use notebook:  [2a_process_html_docling.ipynb](2a_process_html_docling.ipynb)  
or Use python script: [2a_process_html_docling.py](2a_process_html_docling.py)

```bash
python   2a_process_html_docling.py
```

### 2b: Data Prep Kit

Use notebook: [2b_process_html_dpk.ipynb](2b_process_html_dpk.ipynb)  
or Use python script: [2b_process_html_dpk.py](2b_process_html_dpk.py)

```bash
python   2b_process_html_dpk.py
```

## Step-3: Save data into DB

In this step we:

- create chunks from cleaned documents
- create embeddings (embedding models may be downloaded at runtime)
- save the chunks + embeddings into a vector database

We currently use [Milvus](https://milvus.io/) as the vector database.  We use the embedded version, so there is no setup required!


Run the notebook [3_save_to_vector_db.ipynb](3_save_to_vector_db.ipynb)  
or run python script [3_save_to_vector_db.py](3_save_to_vector_db.py)

```bash
python   3_save_to_vector_db.py
```

## Step-4: Query documents

For this step, we need an LLM.

We have two options for running an LLM.

1. Running an open LLM locally using Ollama.  This is FREE and doesn't require any API access (default option)
2. Using a service like [Replicate](https://replicate.com)


### 4.1 - Option 1: local Ollama (default) 

File: [my_config.py](../my_config.py):

```python
MY_CONFIG.LLM_RUN_ENV = 'local_ollama'
MY_CONFIG.LLM_MODEL = "gemma3:1b" 
```

- If you are running the Allycat docker image, then Ollama is already installed.  
- If you running natively on your machine, make sure to install Ollama by following the setup instructions here

Once Ollama is installed, we just need to download the model we are going to be using.

Here is how:

```bash
# download the model
ollama   pull gemma3:1b

# verify the model is available locally
ollama  list
```

Sample output might look like:


### 4.2 - Option 2: For using Replicate

File: [my_config.py](../my_config.py):

```python
MY_CONFIG.LLM_RUN_ENV = 'replicate'
MY_CONFIG.LLM_MODEL = "meta/meta-llama-3-8b-instruct"
```

Also make sure that you have completed [replicate setup](running-natively.md#step-4-replicate-setup-optional)

### Let's run the query

Query documents using LLM

- using notebook [4_query.ipynb](4_query.ipynb)
- or running python script: [4_query.py](4_query.py)

```bash
python  4_query.py
```

## Step-5: Web UI

For this step, we need to run a web interface to interact with our RAG system.

We have two options for the web UI:

1. Flask UI - A simple, lightweight web interface (default option)
2. Chainlit UI - A more feature-rich chat interface with advanced capabilities

### 5.1 - Option 1: Flask UI

```bash
python app.py
```
Go to url : http://localhost:8080  and start chatting!

### 5.2 - Option 2: Chainlit UI

```bash
chainlit run chainlit_app.py --port 8090
```
Go to url : http://localhost:8090  and start chatting!

## Step-6: (Optional) Customizing AllyCat

Please see [customizing-allycat](customizing-allycat.md)