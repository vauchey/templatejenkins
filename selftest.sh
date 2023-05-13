#!/bin/sh
echo "run selftest"
cd /home/user
whoami
ls
pwd
py.test --junitxml results.xml selftest.py
#python3 
