#! /usr/bin/bash
already_run=($(lsof -t -i:6400))
if [ ! -z "$already_run" ]; then
    echo $already_run
    for proc_id in ${already_run[@]}
    do
        echo $proc_id
        kill -9 $proc_id
    done
fi
# pkill gunicorn 

sleep 3 &
wait $!
gunicorn -c conf.py app:app