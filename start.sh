#! /usr/bin/bash
already_run=$(lsof -t -i:6400)
if [ ! -z "$already_run" ]; then
    echo $already_run
    killall -9 "$already_run"
fi

sleep 5 &
wait $!
gunicorn -c conf.py app:app