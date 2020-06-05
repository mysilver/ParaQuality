import logging
import os
import sys
import traceback
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, jsonify, abort
from flask_cors import CORS
from flask_restx import Api, reqparse, Resource
from langid.langid import LanguageIdentifier, model
from numpy import dot
from language_tool import LanguageChecker
from numpy.linalg import norm

lang_checker = LanguageChecker()
language_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
sys.path.append(os.getcwd())
app = Flask(__name__)
CORS(app)
api = Api(app,
          default="APIs",
          version="0.1.1",
          title="ParaQuality",
          description="A service to quality assess crowdsourced paraphrases")

query_parser = reqparse.RequestParser()
query_parser.add_argument('sentence', type=str, help="sentence to be checked")

pair_parser = reqparse.RequestParser()
pair_parser.add_argument('sentence_1', type=str, help="a sentence to be checked")
pair_parser.add_argument('sentence_2', type=str, help="a sentence to be checked")

g = tf.Graph()
with g.as_default(), tf.device('/cpu:0'):
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
    my_result = embed(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()
session = tf.Session(graph=g)
session.run(init_op)


@api.route("/linguistic/check-spelling")
class Spelling(Resource):

    @api.response(200, "Success")
    @api.expect(query_parser)
    def get(self):
        """
        spell checks a given sentence
        """
        args = query_parser.parse_args()
        sentence = args.get("sentence")

        try:
            return jsonify({
                "original": sentence,
                "corrected": lang_checker.spelling_corrector(sentence),
                "errors": [a.__dict__ for a in lang_checker.check(sentence, categories=["TYPOS", "CASING"])]
            })
        except Exception as e:
            abort(501, message="Server is not able to process the request; {}".format(e))
            traceback.print_stack()


@api.route("/linguistic/check-grammar")
class Grammar(Resource):
    @api.response(200, "Success")
    @api.expect(query_parser)
    def get(self):
        """
        spell checks a given sentence
        """
        args = query_parser.parse_args()
        sentence = args.get("sentence")

        try:
            return jsonify({
                "original": sentence,
                "corrected": lang_checker.spelling_corrector(sentence),
                "errors": [a.__dict__ for a in lang_checker.check(sentence, categories=['MISC', 'GRAMMAR'])]
            })
        except Exception as e:
            abort(501, message="Server is not able to process the request; {}".format(e))
            traceback.print_stack()


@api.route("/linguistic/detect-language")
class LanguageDetect(Resource):
    @api.expect(query_parser)
    def get(self):
        """
        detects the language of the given sentence
        """
        try:
            args = query_parser.parse_args()
            sentence = args.get("sentence")
            misspelling = lang_checker.misspellings(sentence)
            lang = "en"

            if len(misspelling) > 1:
                lang, _ = language_identifier.classify(sentence)

            return jsonify({
                "language": lang
            })
        except Exception as e:
            abort(400, message=e)
            traceback.print_stack()


@api.route("/linguistic/semantic-similarity")
class Similarity(Resource):
    @api.expect(pair_parser)
    def get(self):
        def __vec(text):
            embedding = session.run(my_result, feed_dict={text_input: [text]})[0].tolist()
            return embedding

        """
        generates canonical sentences for a list of operations
        """
        try:
            args = pair_parser.parse_args()
            a = __vec(args.get("sentence_1"))
            b = __vec(args.get("sentence_2"))
            return jsonify({
                "USE": dot(a, b) / (norm(a) * norm(b))
            })
        except Exception as e:
            abort(400, message=e)
            traceback.print_stack()


@api.route("/linguistic/cheating")
class Cheating(Resource):
    @api.expect(query_parser)
    def get(self):
        """
        generates canonical sentences for a list of operations
        """
        try:
            return jsonify([])
        except Exception as e:
            abort(400, message=e)
            traceback.print_stack()


if __name__ == '__main__':
    port = 8080

    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    LOGGER = logging.getLogger("artemis.fileman.disk_memoize")
    LOGGER.setLevel(logging.WARN)
    app.run("0.0.0.0", port=port)
