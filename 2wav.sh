#!/bin/bash

mp3="$1"
wav="${mp3/mp3/wav}"

sox "$mp3" -c 1 -r 16000 "$wav"

rm "$mp3"
