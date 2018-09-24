# cartpole

## Docker commands

cartpole:latest

### Build Docker image

docker build -t cartpole .

### Open virtual machine

docker run -it cartpole:latest bash
docker run -it -v $(pwd)/src:/code cartpole:latest bash
