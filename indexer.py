import os
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch


class Pr:
    def __init__(self, alpha):
        self.crawled_folder = Path('/Users/chefthanathip/PycharmProjects/crawled/')
        self.alpha = alpha

    def url_extractor(self):
        url_maps = {}
        all_urls = set([])
        for file in os.listdir(self.crawled_folder):
            if file.endswith(".txt"):
                j = json.load(open(os.path.join(self.crawled_folder, file)))
                all_urls.add(j['url'])
                for s in j['url_lists']:
                    all_urls.add(s)
                url_maps[j['url']] = list(set(j['url_lists']))
        all_urls = list(all_urls)
        return url_maps, all_urls

    def pr_calc(self):
        url_maps, all_urls = self.url_extractor()
        url_matrix = pd.DataFrame(columns=all_urls, index=all_urls)
        for url in url_maps:
            if len(url_maps[url]) > 0 and len(all_urls) > 0:
                url_matrix.loc[url] = (1 - self.alpha) * (1 / len(all_urls))
                url_matrix.loc[url, url_maps[url]] = url_matrix.loc[url, url_maps[url]] + (
                            self.alpha * (1 / len(url_maps[url])))

        url_matrix.loc[url_matrix.isnull().all(axis=1), :] = (1 / len(all_urls))
        x0 = np.matrix([1 / len(all_urls)] * len(all_urls))
        P = np.asmatrix(url_matrix.values)

        prev_Px = x0
        Px = x0 * P
        i = 0
        while (any(abs(np.asarray(prev_Px).flatten() - np.asarray(Px).flatten()) > 1e-8)):
            i += 1
            prev_Px = Px
            Px = Px * P

        print('Converged in {0} iterations: {1}'.format(i, np.around(np.asarray(Px).flatten().astype(float), 5)))
        self.pr_result = pd.DataFrame(Px, columns=url_matrix.index, index=['score']).T.loc[list(url_maps.keys())]


class IndexerWithPR:
    def __init__(self):
        self.crawled_folder = Path('/Users/chefthanathip/PycharmProjects/crawled/')

        with open(self.crawled_folder / 'url_list.pickle', 'rb') as f:
            self.file_mapper = pickle.load(f)

        self.es_client = Elasticsearch("https://localhost:9200", basic_auth=("elastic", "elastic123"),
                                       verify_certs=False)
        self.pr = Pr(alpha=0.85)

    def run_indexer(self):
        self.pr.pr_calc()

        self.es_client.options(ignore_status=[400, 404]).indices.delete(index='simple')
        self.es_client.options(ignore_status=[400]).indices.create(index='simple')

        self.es_client.options(ignore_status=[400, 404]).indices.delete(index='custom')
        index_body = {
            "settings": {
                "similarity": {
                    "custom_similarity": {
                        "type": "scripted",
                        "source": "double idf = Math.log((field.docCount + 1.0) / (term.docFreq + 1.0)) + 1.0; return idf;"
                    }
                }
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text", "similarity": "custom_similarity"}
                }
            }
        }
        self.es_client.options(ignore_status=[400]).indices.create(index='custom', body=index_body)

        for file in os.listdir(self.crawled_folder):
            if file.endswith(".txt"):
                j = json.load(open(os.path.join(self.crawled_folder, file)))
                j['id'] = j['url']
                j['pagerank'] = self.pr.pr_result.loc[j['id']].score

                self.es_client.index(index='simple', body=j)
                self.es_client.index(index='custom', body=j)
                print(f"Indexed: {j['url']}")


if __name__ == '__main__':
    s = IndexerWithPR()
    s.run_indexer()