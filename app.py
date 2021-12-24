#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from config import get_cfg
from flask import Flask, jsonify, request
from qa.matching.semantic.semantic_similarity import SemanticSimilarity
from qa.queryUnderstanding.querySegmentation.segmentation import Segmentation
from qa.tools import setup_logger
from qa.main import Search
from qa.intent import Fasttext
from qa.knowledge import EntityLink
from qa.search import SearchHelper
from qa.queryUnderstanding.queryReformat.queryCorrection.correct import SpellCorrection

logger = setup_logger(name='APP')
cfg = get_cfg()

intent = Fasttext(cfg, 'two_intent')
searchObj = Search(cfg)
el = EntityLink(cfg)
helper = SearchHelper(cfg)
sc = SpellCorrection(cfg)
ss = SemanticSimilarity(cfg)
seg = Segmentation(cfg)
logger.info("Success load model!!!")

app = Flask(__name__)


@app.route('/intent', methods=["POST"])
def index():
    data = request.json
    query = data['query']
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


@app.route('/hot', methods=["POST"])
def hot():
    hot_question = helper.hot_query()
    return jsonify({"hot_question": hot_question})


@app.route('/spam_detect', methods=["POST"])
def spam_detect():
    data = request.json
    query = data['query']
    flag, sensetive_words = helper.sensetive_detect(query)
    return jsonify({"flag": flag, "sensetive_words": sensetive_words})


@app.route('/spell_correct', methods=["POST"])
def spell_correct():
    data = request.json
    query = data['query']
    e_pos, can, max_score = sc.correct(query)
    return jsonify({
        "error_pos": e_pos,
        "candidate": can,
        'error_score': max_score
    })


@app.route('/term_weight', methods=["POST"])
def term_weight():
    data = request.json
    query = data['query']
    rough_query = seg.cut(query, is_rough=True)
    segmentation = rough_query[0]
    tw = ss.delete_diff(segmentation).tolist()
    return jsonify({
        "segmentation": segmentation,
        'term_weight': tw
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=cfg.WEB.PORT, debug=True)