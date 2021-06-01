#!/usr/bin/python
import sys
import site

site.addsitedir('/home/ubuntu/.local/lib/python3.8/site-packages')
sys.path.insert(0, '/var/www/html/workout-app/')

from flaskapp import app as application