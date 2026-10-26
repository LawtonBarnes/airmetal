#!/bin/bash
# Runs a dev copy of AIR METAL on PRODUCTION's wall, outside STRINGS.
#   wall_test.sh start DIR [args]   stop STRINGS, launch DIR/main.py on tty1
#   wall_test.sh deploy TGZ [args]  replace /tmp/amdev/airmetal from TGZ, (re)start it
#   wall_test.sh stop               kill it, give the wall back to STRINGS
# Lives in a file on purpose: a `pkill -f main.py` typed into `ssh pr '...'`
# also matches (and kills) that ssh command's own shell.
set -u
LOG=/tmp/airmetal_wall.log

kill_app() {
    sudo pkill -TERM -f "^python3 (-B )?main\.py" ; sleep 2
    sudo pkill -KILL -f "^python3 (-B )?main\.py"
}

launch() {
    dir="$1"; shift
    sudo rm -f "$LOG"
    sudo openvt -f -c 1 -s -- sh -c "cd '$dir' && exec python3 -B main.py $* > $LOG 2>&1"
}

case "$1" in
start)
    dir="$2"; shift 2
    sudo timeout 20 systemctl stop strings || sudo systemctl kill -s KILL strings
    sleep 1
    launch "$dir" "$@"
    ;;
deploy)
    tgz="$2"; shift 2
    kill_app
    sudo rm -rf /tmp/amdev/airmetal
    mkdir -p /tmp/amdev && tar xzf "$tgz" -C /tmp/amdev
    launch /tmp/amdev/airmetal "$@"
    ;;
stop)
    kill_app
    sudo systemctl start strings
    ;;
esac
