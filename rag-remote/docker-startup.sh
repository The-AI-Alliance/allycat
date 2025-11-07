#!/bin/bash

if [ "$1" == "deploy" ]; then
  echo "In deploy mode..."

  echo "Starting web server..."
  # run the web server in foreground so the container doesn't exit
  # python3 app_flask.py 
  chainlit run app_chainlit.py --host 0.0.0.0 --port 8090 --headless
  # python3 app_flask.py > /allycat/app.out 2>&1 &
  # # wait for the web server to start
  # while ! nc -z localhost 8080; do
  #   # cat /allycat/app.out
  #   sleep 1
  # done
  # echo "Web server started on port 8080"
else
  echo "Not in deploy mode, skipping  web server start."
  echo "You can run the following commands to start the web server:"
  # echo "      python3 app_flask.py"
  echo "      chainlit run app_chainlit.py --port 8090"
  echo "Then open your browser and go to http://localhost:8090"
  echo "To stop the web server, run: killall python3"
  /bin/bash
fi
