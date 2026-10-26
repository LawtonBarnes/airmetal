#!/bin/bash
# Runs a dev copy of AIR METAL on PRODUCTION's wall, outside STRINGS.
#   wall_test.sh start DIR [args]   stop STRINGS, launch DIR/main.py on tty1
#   wall_test.sh stop               kill it, give the wall back to STRINGS
set -u
case "$1" in
start)
    dir="$2"; shift 2
    sudo timeout 20 systemctl stop strings || sudo systemctl kill -s KILL strings
    sleep 1
    sudo openvt -f -c 1 -s -- sh -c "cd '$dir' && exec python3 main.py $* > /tmp/airmetal_wall.log 2>&1"
    ;;
stop)
    sudo pkill -TERM -f "^python3 main.py" ; sleep 2
    sudo pkill -KILL -f "^python3 main.py"
    sudo systemctl start strings
    ;;
esac
