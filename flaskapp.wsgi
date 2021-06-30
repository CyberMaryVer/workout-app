#!/usr/bin/python
import sys
import site

ALLDIRS = ['/home/ubuntu/venv/lib/python3.8/site-packages']
prev_sys_path = list(sys.path)
for directory in ALLDIRS:
    site.addsitedir(directory)

new_sys_path = []
for item in list(sys.path):
    if item not in prev_sys_path:
        new_sys_path.append(item)
        sys.path.remove(item)

sys.path[:0] = new_sys_path
activate_this = '/home/ubuntu/venv/bin/activate_this.py'
execfile(activate_this, dict(__file__=activate_this))

# WSGIPythonPath /home/ubuntu/venv/lib/python3.8/site-packages
# /usr/local/lib/python3.8/dist-packages
# WSGIDaemonProcess flaskapp \ python-path=home/ubuntu/venv/lib/python3.8/site-packages

# site.addsitedir('/home/ubuntu/venv/lib/python3.8/site-packages')
# site.addsitedir('/home/ubuntu/.local/lib/python3.8/dist-packages')
# sys.path.insert(0, '/var/www/html/workout-app/')

from flaskapp import app as application