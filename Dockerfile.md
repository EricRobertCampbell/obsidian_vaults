Instructions for building the image

- FROM: existing image to base the new one on
- WORKDIR: Set the working directory in the container (also moves here)
- COPY <outside> <inside>: copy the files <outside> internally to the container <inside>
- RUN: run a shell command
- CMD: default command to run when running the container
- EXPOSE: expose the given port (internal to the container)