#!/bin/bash
# Budget guard: the machine powers itself off 15 h after first boot (instance-initiated shutdown = STOP, disk kept).
shutdown -h +900 "ML challenge budget guard (15 h)"
dnf install -y rsync tmux htop git gcc gcc-c++ >/var/log/ml-setup.log 2>&1
