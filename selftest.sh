#!/bin/sh
echo "run selftest"
cd /home/user
sudo chown -R user /home/user
whoami
ls
pwd
#py.test --junitxml results.xml selftest.py
pip install -r ./requirements.txt
python3 selftest.py
ls
#python3 
#je fait un docker (environement minimal, puis je lance un pip install requirement dans un venv et lance le test)
