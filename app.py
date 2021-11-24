#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from config import get_cfg
from flask import Flask, jsonify, request
from qa.tools import setup_logger
from qa.main import Search
from qa.intent import Fasttest
from qa.knowledge import EntityLink

logger = setup_logger()
cfg = get_cfg()

intent = Fasttest(cfg, 'two_intent')
searchObj = Search(cfg)
el = EntityLink(cfg)

app = Flask(__name__)


@app.route('/intent', methods=["POST"])
def index():
    data = request.json
    query = data['query']
    query = query if isinstance(query, list) else [query]
    prediction, proba = intent.predict(query)
    return jsonify({"label": prediction, 'score': proba})


@app.route('/retrieval', methods=["POST"])
def retrieval():
    data = request.json
    query = data['query']
    result = searchObj.search(query)
    return jsonify({"result": result})


@app.route('/entity_link', methods=["POST"])
def entity():
    data = request.json
    query = data['query']
    entity, type = el.entity_link(query)
    return jsonify({"entity": entity, "type": type})
0

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=cfg.WEB.PORT, debug=True)