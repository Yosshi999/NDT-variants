#!/bin/sh

touch .env
echo "uid=$(id -u)" >> .env
echo "gid=$(id -g)" >> .env

