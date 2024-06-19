from flask import Flask, Response, render_template, request, jsonify
from data import get_coords


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def data():
    line_id = request.args.get("line_id")
    time = request.args.get("time")

    if not line_id or not time:
        return Response(
            "line_id or time parameter was empty",
            status="404",
        )

    if not line_id.isdigit() or not time.isdigit():
        return Response(
            "line_id or time parameter was not a valid integer",
            status="404"
        )

    return jsonify(get_coords(int(line_id), int(time)))
