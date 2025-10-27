#!/bin/bash

## define OLLAMA_MODELS dir
if [ -z "$OLLAMA_MODELS" ]; then
  export OLLAMA_MODELS=/allycat/workspace/ollama
fi
# export OLLAMA_HOST="0.0.0.0:11434"
echo "Env variables for OLLAMA"
env | grep OLLAMA

# start ollama
# ollama_model="qwen3:1.7b"
ollama_model="gemma3:1b"
echo "Starting Ollama..."
ollama serve   > /allycat/ollama-serve.out  2>&1 &
# wait for ollama to start
while ! nc -z localhost 11434; do
#   cat /allycat/ollama-serve.out
  sleep 1
done
echo "Ollama started on port 11434"

# only download the model if we are in DEPLOY mode
if [ "$1" == "deploy" ]; then
  echo "In deploy mode..."
  echo "Downloading model $ollama_model ..."
  ollama pull $ollama_model

  echo "Starting web server..."
  # run the web server in foreground so the container doesn't exit
  python3 app_flask.py 
  # python3 app_flask.py > /allycat/app.out 2>&1 &
  # # wait for the web server to start
  # while ! nc -z localhost 8080; do
  #   # cat /allycat/app.out
  #   sleep 1
  # done
  # echo "Web server started on port 8080"
else
  echo "Not in deploy mode, skipping model download and web server start."
  echo "To download the model, run: "
  echo "      ollama pull $ollama_model"
  echo "You can run the following commands to start the web server:"
  echo "      python3 app_flask.py"
  echo "Then open your browser and go to http://localhost:8080"
  echo "To stop the web server, run: killall python3"
  echo "To stop the ollama server, run: killall ollama"
  /bin/bash
fi
