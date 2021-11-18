#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from config import get_cfg
from flask import Flask, jsonify, request
from core.tools import setup_logger
from core.main import Search
from core.intent import Fasttest
from core.knowledge import EntityLink

logger = setup_logger()
cfg = get_cfg

intent = Fasttest(cfg, 'two_intent')
searchObj = Search(cfg)
el = EntityLink(cfg)

app = Flask(__name__)


@app.route('/intent', methods=["POST"])
def index():
    query = request.json
    query = query if isinstance(query, list) else [query]
    prediction, proba = intent.predict(query)
    return jsonify({"label": prediction, 'score': proba})


@app.route('/retrieval', methods=["POST"])
def index():
    query = request.json
    result = searchObj.search(query)
    return jsonify({"result": result})


@app.route('/entity_link', methods=["POST"])
def index():
    query = request.json
    entity = el.entity_link(query)
    return jsonify({"entity": entity})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=cfg.WEB.PORT, debug=True)