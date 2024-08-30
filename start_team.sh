#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 arg1"
    exit 1
fi

ARG1=$1

for ((i=1; i<=11; i++))
do
    python mujoco_client.py localhost 60000 $ARG1 $i &
done