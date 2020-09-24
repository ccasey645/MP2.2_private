import math
import sys
import time

import metapy
import pytoml
from scipy import stats

class InL2Ranker(metapy.index.RankingFunction):
    """
    Create a new ranking function in Python that can be used in MeTA.
    """
    def __init__(self, param=1.0):
        self.param = param
        # You *must* call the base class constructor here!
        super(InL2Ranker, self).__init__()

    def score_one(self, sd):
        """
        You need to override this function to return a score for a single term.
        For fields available in the score_data sd object,
        @see https://meta-toolkit.org/doxygen/structmeta_1_1index_1_1score__data.html
        """
        normalized_doc_length = 1.0 + (sd.avg_dl / sd.doc_size)
        tfn = sd.doc_term_count * math.log(normalized_doc_length, 2)
        first_half = sd.query_term_weight * (tfn / (tfn + self.param))
        middle = (sd.num_docs + 1) / (sd.corpus_term_count + 0.5)
        second_half = math.log(middle,2)
        return first_half * second_half


def load_ranker(cfg_file, param=1.0):
    """
    Use this function to return the Ranker object to evaluate, e.g. return InL2Ranker(some_param=1.0) 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index. You can ignore this for MP2.
    """
    return InL2Ranker(param=param)


def run_queries(idx, ranker, ev):
    start_time = time.time()
    top_k = 10
    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)
    query = metapy.index.Document()
    avg_precisions = []
    print('Running queries')
    with open(query_path) as query_file:
        for query_num, line in enumerate(query_file):
            query.content(line.strip())
            results = ranker.score(idx, query, top_k)
            avg_p = ev.avg_p(results, query_start + query_num, top_k)
            print("Query {} average precision: {}".format(query_num + 1, avg_p))
            avg_precisions.append(avg_p)
    print("Mean average precision: {}".format(ev.map()))
    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))
    return avg_precisions

def write_list_to_file(file_name, values):
    with open(file_name, "w") as f:
        for item in values:
            f.write("%s\n" % item)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    cfg = sys.argv[1]
    print('Building or loading index...')
    idx = metapy.index.make_inverted_index(cfg)
    try:
        ranker_param = float(sys.argv[2])
    except IndexError:
        ranker_param = 5.7

    print("printing ranker_param: {}".format(ranker_param))
    ranker = load_ranker(cfg, ranker_param)
    ev = metapy.index.IREval(cfg)

    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)

    query_cfg = cfg_d['query-runner']
    if query_cfg is None:
        print("query-runner table needed in {}".format(cfg))
        sys.exit(1)

    avg_precisions = run_queries(idx, ranker, ev)
    inl2_file_name = "inl2.avg_p.txt"
    write_list_to_file(inl2_file_name, avg_precisions)

    # #Change to BM25
    ranker = metapy.index.OkapiBM25()
    bm_25_avg_precisions = run_queries(idx, ranker, ev)
    bm25_file_name = "bm25.avg_p.txt"
    write_list_to_file(bm25_file_name, bm_25_avg_precisions)

    #Compare two precisions
    t_statistic, p_value = stats.ttest_rel(avg_precisions, bm_25_avg_precisions)
    print("significance!!!! {}".format(p_value))
    write_list_to_file("significance.txt", [p_value])
