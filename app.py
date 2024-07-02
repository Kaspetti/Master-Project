from flask import Flask, Response, render_template, request, jsonify
from data import get_coords, get_line_amount


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/coords")
def coords():
    ens_id = request.args.get("ens-id")
    line_id = request.args.get("line-id")
    time = request.args.get("time")

    if not line_id or not time or not ens_id:
        return Response(
            "ens-id, line-id, or time parameter was empty",
            status="404",
        )

    if not line_id.isdigit() or not time.isdigit() or not ens_id.isdigit():
        return Response(
            "ens-id, line-id, or time parameter was not a valid integer",
            status="404"
        )

    return jsonify(get_coords(int(ens_id), int(line_id), int(time)))


@app.route("/api/line-count")
def line_count():
    ens_id = request.args.get("ens-id")
    time = request.args.get("time")

    if not time or not ens_id:
        return Response(
            "ens-id or time parameter was empty",
            status="404",
        )

    if not time.isdigit() or not ens_id.isdigit():
        return Response(
            "ens-id or time parameter was not a valid integer",
            status="404"
        )

    return jsonify(get_line_amount(int(ens_id), int(time)))
