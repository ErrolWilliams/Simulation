import os
packages = ['pandas','keras','tensorflow']

os.system('mkdir models')
os.system('sudo apt update')
os.system('sudo apt install python3-pip')

for package in packages:
    os.system('pip3 install {0}'.format(package))
