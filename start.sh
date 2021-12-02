#! /usr/bin/bash
kill $(lsof -t -i :8080)
gunicorn -c conf.py app:app