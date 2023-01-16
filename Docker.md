# Docker
[Homepage](https://www.docker.com/)
- Seems to bundle together everything that an application needs to run into a container, which reduces the reliance on the actual computer running it
	- (Dependencies live in the container, so the environment is always the same)
- [[Dockerfile]] contains instructions for building a [[Docker Image]]
- A [[Docker Container]] is a runnable instance of an image
- One advantage is that the idea behind them is to be able to kill and spin up new instances of the container relatively quickly when e.g. load balancing
- You can start with a known image and then add additional information / processes as a new layer
- [[Kubernetes]] is used to orchestrate multiple [[Docker Image]]s: