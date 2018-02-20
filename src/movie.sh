#!/bin/sh
# 
# Creates movie file based on sequential images.
#
# Usage: movie.sh <fps> <image extension> <output file basename>
# 	e.g. movie.sh 5 png movie
#
# Requirements:
# 
# • mencoder
# • ffmpeg
#

bitrate=900

# creates sequential list of files that will be rendered
ls plot.*.$2 | sort -n -t . -k 2 > list.txt

mencoder "mf://@list.txt" -mf fps=$1 -o $3.avi -ovc lavc -lavcopts vcodec=msmpeg4v2:vbitrate=$bitrate

# Quicktime compatible movie (OS X)
# ===========================
# http://www.mplayerhq.hu/DOCS/HTML/en/menc-feat-quicktime-7.html

# pass 1
#mencoder "mf://@list.txt" -ovc x264 -x264encopts pass=1:turbo:bitrate=$bitrate:bframes=1:me=umh:partitions=all:trellis=1:qp_step=4:qcomp=0.7:direct_pred=auto:keyint=300:threads=auto -oac faac -faacopts br=192:mpeg=4:object=2 -channels 2 -srate 48000 -mf fps=$1 

# pass 2
#mencoder "mf://@list.txt" -o movie.avi -ovc x264 -x264encopts pass=2:turbo:bitrate=$bitrate:frameref=5:bframes=1:me=umh:partitions=all:trellis=1:qp_step=4:qcomp=0.7:direct_pred=auto:keyint=300:threads=auto -oac faac -faacopts br=192:mpeg=4:object=2 -channels 2 -srate 48000 -mf fps=$1 

ffmpeg -i $3.avi -acodec libmp3lame -ab 192 movie.mov