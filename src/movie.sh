#!/bin/sh
# 
# Creates movie file based on sequential images in the current directory.
#
# Usage: movie.sh <image extension> <fps> <output file basename>
#
# Example:
#
# $ movie.sh png 5 movie
#
# 	will create a movie from all *.png files in the directory, at 5 frames
# 	per second, and create a file movie.mov.
#
# Requirements:
# 
# • mencoder
# • ffmpeg
#

bitrate=900

# creates sequential list of files that will be rendered
ls *.$1 | sort -n -t . -k 2 > list.txt

mencoder "mf://@list.txt" -mf fps=$2 -o $3.avi -ovc lavc -lavcopts vcodec=msmpeg4v2:vbitrate=$bitrate

# Quicktime compatible movie (MacOS)
# ====================================
# http://www.mplayerhq.hu/DOCS/HTML/en/menc-feat-quicktime-7.html

# pass 1
#mencoder "mf://@list.txt" -ovc x264 -x264encopts pass=1:turbo:bitrate=$bitrate:bframes=1:me=umh:partitions=all:trellis=1:qp_step=4:qcomp=0.7:direct_pred=auto:keyint=300:threads=auto -oac faac -faacopts br=192:mpeg=4:object=2 -channels 2 -srate 48000 -mf fps=$1 

# pass 2
#mencoder "mf://@list.txt" -o movie.avi -ovc x264 -x264encopts pass=2:turbo:bitrate=$bitrate:frameref=5:bframes=1:me=umh:partitions=all:trellis=1:qp_step=4:qcomp=0.7:direct_pred=auto:keyint=300:threads=auto -oac faac -faacopts br=192:mpeg=4:object=2 -channels 2 -srate 48000 -mf fps=$1 

ffmpeg -i $3.avi -acodec libmp3lame -ab 192 movie.mov