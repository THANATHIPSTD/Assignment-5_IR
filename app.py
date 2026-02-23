from flask import Flask, request, render_template, jsonify
from elasticsearch import Elasticsearch
import pandas as pd
import time
import re

app = Flask(__name__)
app.es_client = Elasticsearch("https://localhost:9200", basic_auth=("elastic", "elastic123"), verify_certs=False)


def format_snippet(text, query):
    if not text: return ""
    sentences = re.split(r'(?<=[.!?]) +|\n+', text)
    snippet_sentences = []

    for i, s in enumerate(sentences):
        if query.lower() in s.lower():
            start = max(0, i - 1)
            end = min(len(sentences), i + 2)
            snippet_sentences = sentences[start:end]
            break

    if not snippet_sentences:
        snippet_sentences = sentences[:2]

    snippet = " ".join(snippet_sentences)
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    snippet = pattern.sub(lambda m: f"<b>{m.group(0)}</b>", snippet)
    return snippet


def perform_search(index_name, query_term):
    start = time.time()
    response_object = {'status': 'success'}

    results = app.es_client.search(
        index=index_name,
        source_excludes=['url_lists'],
        size=100,
        query={
            "script_score": {
                "query": {"match": {"text": query_term}},
                "script": {"source": "_score * doc['pagerank'].value"}
            }
        }
    )

    end = time.time()
    total_hit = results['hits']['total']['value']

    formatted_results = []
    for hit in results['hits']['hits']:
        title = hit["_source"].get('title', 'No Title')
        url = hit["_source"].get('url', '#')
        raw_text = hit["_source"].get('text', '')
        score = hit["_score"]

        snippet = format_snippet(raw_text, query_term)

        formatted_results.append({
            'title': title,
            'url': url,
            'text': snippet,
            'score': score
        })

    response_object['total_hit'] = total_hit
    response_object['results'] = formatted_results
    response_object['elapse'] = round(end - start, 4)
    return response_object


@app.route('/')
def home():
    query = request.args.get('q', '')
    return render_template('index.html', query=query)


@app.route('/api/search_bm25')
def search_bm25():
    query = request.args.get('q', '')
    return jsonify(perform_search('simple', query))


@app.route('/api/search_tfidf')
def search_tfidf():
    query = request.args.get('q', '')
    return jsonify(perform_search('custom', query))


if __name__ == '__main__':
    app.run(debug=True)