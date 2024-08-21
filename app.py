from flask import Flask, Response, render_template, request, jsonify
from data import get_all_lines


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/all-lines")
def all_lines():
    time = request.args.get("date")

    if not time:
        return Response(
            "time parameter empty",
            status="400",
        )

    if not time.isdigit():
        return Response(
            "time is not a valid integer",
            status="400"
        )

    return jsonify(get_all_lines(int(time)))
