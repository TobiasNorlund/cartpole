# cartpole

## Docker commands

https://github.com/jaimeps/docker-rl-gym
cartpole:latest

### Build Docker image

docker build -t cartpole .

### Open virtual machine

docker run -it cartpole:latest bash
docker run -it -v $(pwd)/src:/code cartpole:latest bash

docker run -it -v $(pwd)/src:/code -p 8888:8888 cartpole:latest jupyter lab --ip 0.0.0.0 --allow-root
