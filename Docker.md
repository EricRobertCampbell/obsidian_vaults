# Docker
[Homepage](https://www.docker.com/)
- Seems to bundle together everything that an application needs to run into a container, which reduces the reliance on the actual computer running it
	- (Dependencies live in the container, so the environment is always the same)
- [[Dockerfile]] contains instructions for building a [[Docker Image]]
- A [[Docker Container]] is a runnable instance of an image
- One advantage is that the idea behind them is to be able to kill and spin up new instances of the container relatively quickly when e.g. load balancing
- You can start with a known image and then add additional information / processes as a new layer
- [[Kubernetes]] is used to orchestrate multiple [[Docker Image]]s (similar to [[docker compose]])
- Steps to running a [[Docker Image]] (from [StereoLabsl](https://www.stereolabs.com/docs/docker/creating-your-image/)):
	1. Write the [[Dockerfile]] for your application
	2. Build the image with the `docker build` command
	3. Host the image on a registry
	4. Pull and run the image on the target machine
 - [[Getting Started - Docker Tutorial]]
 - Store the docker images on a repository ([Docker Hub](https://hub.docker.com/))
 - By default, files created in one container are not visible to other containers, *even if* they are built usig the same image
 - You can use [[Docker Container Volumes|container volumes]] to change this
	 - Volumes provide the ability to connect specific filesstem paths in the host machine to specific ones in the host machine. If files change there, then all containers with access to that file will see the change
	 - There are two main kinds of 
  

## Commands
- `docker build -t getting-started .`:  build the image from the [[Dockerfile]] in the current directory. The `-t` flag indicates the human-readable name of the image (`getting-started`)
- `docker run -d -p <outside port>:<container port> getting-started`: 
	- Run the image named `getting-started`
	- `-d`: run in detached mode (as a daemon)
	- `-p <outside port>:<inside port>`: map the `inside` port within the container to the `outside` port outside the container
- `docker ps` - list the running docker images
- `docker stop <id>` - stop the running container with `id`
- `docker rm <id>` - remove the container with `id`
- `docker rm -f <id>` - stop and remove the container in one command (the `-f` flag is `force`)
- `docker push ericrobertcampbell/getting-started:tagname` - push the existing image in the current directory with name `getting-started` to [[Docker Hub]] with the tag `tagname`
- `docker login -u <username>`: login with the given username
- `docker tag getting-started <username>/getting-started`: tag the existing image `getting-started` with one on Docker Hub
- `docker exec <container id> <command>`: execute `command` inside the container with `<container id>`



 
