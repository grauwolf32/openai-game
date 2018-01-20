#!/bin/sh
if  [ -n "$1" ]
then
    rm -rf ./log
    mkdir log
    export OPENAI_LOGDIR=`pwd`/log
    export OPENAI_LOG_FORMAT=tensorboard,stdout
    python3 run_acktr.py --fname=`pwd`/model/$1.ckpt --env=$1
else
    printf "Usage: run_command.sh env_name"
fi