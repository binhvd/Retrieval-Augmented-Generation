# Installation

### Download and setup an Ubuntu container
docker run -p 8501:8501 --name rag -it -d ubuntu:22.04

### Log into the container
docker exec -it rag bash

### Install Vim, Python3 and pip
apt update
apt install vim python3 python3-pip

### Install Python libraries
pip install -r requirements.txt

### Add OpenAI API Key
vim App.py

### Run the app
streamlit run App.py
