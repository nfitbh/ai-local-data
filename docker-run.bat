docker build . -t ai-data-docker
docker run -e OPENAI_API_KEY=%OPENAI_API_KEY% --rm -it --name ai-data ai-data-docker

