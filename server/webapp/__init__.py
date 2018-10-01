from flask import Flask

app = Flask(__name__)

from webapp.views import common
