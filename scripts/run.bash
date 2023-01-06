#!/usr/bin/env bash

if [ "$DOCKER_RUNNING" == true ] 
then
    echo "Inside docker instance, I don't know why you'd want to nest terminals?"
    exit 1
    
else
    echo "To start up the environment, type this 'source activate opencvconda'"

    
    # echo "Starting up docker instance..."

    # cmp_volumes="--volume=$(pwd):/app/:rw"

    # docker run --rm -ti \
    #     $cmp_volumes \
    #     -it \
    #     -e DISPLAY=$DISPLAY \
    #     -v /tmp/.X11-unix:/tmp/.X11-unix \
    #     --device=/dev/video0:/dev/video0 \
    #     --ipc host \
    #     imagename:latest \
    #     /bin/bash
    #--device=/dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
fi