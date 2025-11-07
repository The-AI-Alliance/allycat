# Allycat RAG Remote

This setup  runs Allycat RAG - runs models on cloud services.

## Prerequisites

- API_KEYs for service we wll be using.  For example, to use Nebius AI, we will need `NEBIUS_API_KEY`

## Tech Stack

| Component        | Functionality | Runtime | 
|------------------|---------------|----|
| [Milvus](https://milvus.io/) embedded | Vector db     | Locally or remotely | 
| Models          | LLM runtime   | Remotely (Nebius, Replicate ...etc) | 


## Step-1: Get the code

```bash
# Substitute appropriate repo URL
git   clone https://github.com/The-AI-Alliance/allycat/
cd    allycat/rag-remote
```

---

## Step-2: Setup 

Follow the [Setup](setup.md) guide.

**And activate your python env**

```bash
## if using uv
source .venv/bin/activate

## if using python venv
source  .venv/bin/activate

## If using conda
conda  activate  allycat-1  # what ever the name of the env
```

---

## Step-3: Setup `.env` file


A sample `env.sample.txt` is provided.  Copy this file into `.env` file.

```bash
cp  env.sample.txt  .env
```

And edit `.env` file to make your changes.

**1) To use Nebius AI**

- Get NEBIUS_API_KEY from [Nebius](https://tokenfactory.nebius.com/)
- add the NEBIUS_API_KEY to `.env` file

```text
NEBIUS_API_KEY = "your key goes here" 
```

The default models used will be:
- LLM: `nebius/Qwen/Qwen3-30B-A3B-Instruct-2507`
- Embedding: `nebius/Qwen/Qwen3-Embedding-8B`

**Optionally** you can configure models we might use in `.env` file:

Find the available models at [Nebius Token Factory](https://tokenfactory.nebius.com/)

```text
EMBEDDING_MODEL = Qwen/Qwen3-Embedding-8B
EMBEDDING_LENGTH = 384

LLM_MODEL = nebius/Qwen/Qwen3-30B-A3B-Instruct-2507
```

---

## Allycat Workflow

![](../assets/rag-website-1.png)

## Step-4: Crawl the website


This step will crawl a site and download the website content into the `workspace/crawled` directory

code: [1_crawl_site.py](1_crawl_site.py)


```bash
# default settings
python     1_crawl_site.py  --url https://thealliance.ai

## if using uv
# uv run python     1_crawl_site.py  --url https://thealliance.ai

# or specify parameters
python  1_crawl_site.py   --url https://thealliance.ai --max-downloads 100 --depth 5
# uv run python  1_crawl_site.py   --url https://thealliance.ai --max-downloads 100 --depth 5
```

## Step-5: Process Downloaded files

We will process the downloaded files (html / pdf) and extract the text as markdown.  The output will be saved in the`workspace/processed` directory in markdown format

We use [Docling](https://github.com/docling-project/docling) to process downloaded files.  It will convert the files into markdown format for easy digestion.

- Use python script: [2_process_files.py](2_process_files.py)
- or (For debugging) Run notebook :  [2_process_files.ipynb](2_process_files.ipynb)  

```bash
python   2_process_files.py
# uv run python   2_process_files.py
```

---

## Step-6: Save data into Milvus Vector DB

In this step we:

- create chunks from cleaned documents
- create embeddings (embedding models may be downloaded at runtime)
- save the chunks + embeddings into a vector database

We currently use [Milvus](https://milvus.io/) as the vector database.  We use the embedded version, so there is no setup required!


- Run python script [3_save_to_vector_db.py](3_save_to_vector_db.py)
- or (For debugging) Run the notebook [3_save_to_vector_db.ipynb](3_save_to_vector_db.ipynb)  

```bash
python   3_save_to_vector_db.py
```

---

## Step-7: Query documents

- running python script: [4_query.py](4_query.py)
- or (for debug) using notebook [4_query.ipynb](4_query.ipynb)

```bash
python  4_query.py
```

## Step-8: Run Web UI

**Option 1: Flask UI**

```bash
python app_flask.py
```

Go to url : http://localhost:8080  and start chatting!

**Option 2: Chainlit UI**

```bash
chainlit run app_chainlit.py --port 8090
```

Go to url : http://localhost:8090  and start chatting!

---

## Packaging the app to deploy

We will create a docker image of the app.  It will package up the code + data

Note:  Be sure to run the docker command from the root of the project.

```bash
docker  build    -t allycat-remote  .
```

## Run the AllyCat Docker

Let's start the docker in 'dev' mode

```bash
docker run -it --rm -p 8090:8090  -p 8080:8080  allycat-remote  deploy
# docker run -it --rm -p 8090:8090  -v allycat-vol1:/allycat/workspace  sujee/allycat
```

`deploy` option starts web UI.


---

## Dev Notes

### Creating `requirements.txt` using uv

when uv dependencies are updated, run this command to create requirements.txt

```bash
uv export --frozen --no-hashes --no-emit-project --no-default-groups --output-file=requirements.txt
```