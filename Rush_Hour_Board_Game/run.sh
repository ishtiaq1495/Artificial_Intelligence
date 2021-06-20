#!/bin/sh
if [ "$#" -gt 1 ]; then
	python rushhour.py "$1" "$2"
else
	python rushhour.py "$1"
fi