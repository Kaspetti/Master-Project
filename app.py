from flask import Flask, Response, render_template, jsonify
from main import integrate_flow
import json
import numpy 

app = Flask(__name__)

solutions = integrate_flow(300)

data = [
    [[(x, y) for x in s[0]] for y in s[1]]
    for s in solutions
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ensemble")
def ensemble():
    data = [
            [[(x, y) for x in s[0]] for y in s[1]]
            for s in solutions
            ]
    return jsonify(data),
