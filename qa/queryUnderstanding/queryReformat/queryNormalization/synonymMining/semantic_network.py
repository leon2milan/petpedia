import networkx as nx
import codecs
from multiprocessing import Pool, Process, Lock
from qa.tools import setup_logger
import os

logger = setup_logger()


def multiprocess_network(path, win_len=3, word2id=None):
    logger.info(
        'multi-process construct semantic network ...... path = {a}'.format(
            a=path))

    p = Pool()
    for rt, dirs, files in os.walk(path):
        for f in files:
            file = os.path.join(path, f)
            logger.info('process file {a}'.format(a=file))
            args = (file, word2id, win_len)
            p.apply_async(generate_network_from_corpus, args=(args, ))

        logger.info('wait for all process.....')
        p.close()
        p.join()
        logger.info('done!!!.')


def generate_network_from_corpus(args):

    data, word2id, win_len, base_path = args
    edge_dict = {}
    word_freq_dict = {}

    for sent_list in data:

        # Word frequency
        for word in sent_list:
            if word not in word2id:
                continue
            if word not in word_freq_dict:
                word_freq_dict[word] = 0
            word_freq_dict[word] += 1

        # Process each line
        for i in range(len(sent_list)):
            # window size  [-win_len,+win_len], total 2*win_len+1
            start = max(0, i - win_len)
            end = min(len(sent_list), i + win_len + 1)
            for index in range(start, end):
                if index == i:
                    continue
                else:
                    score = win_len - abs(index - i) + 1
                    node1 = sent_list[index]
                    node2 = sent_list[i]
                    if node1 in word2id.keys() and node2 in word2id.keys():
                        if node1 < node2:
                            edge = (node1, node2)
                        else:
                            edge = (node2, node1)

                        if edge in edge_dict:
                            edge_dict[edge] += score
                        else:
                            edge_dict[edge] = score
    logger.info("start stat word frequency...".format(a=len(edge_dict)))
    w_f_line = ''
    for (k, v) in word_freq_dict.items():
        w_f_line += k + ',' + str(v) + '\n'

    with open(os.path.join(base_path, 'word_frequent.txt'), 'w', encoding='utf8') as f:
        f.write(w_f_line)

    logger.info(" write network to file，total number of edges = {a} ".format(
        a=len(edge_dict)))
    line = ""
    for (node1, node2) in edge_dict:
        id1 = word2id.get(node1)
        id2 = word2id.get(node2)
        weight = edge_dict.get((node1, node2))
        line += node1 + ',' + node2 + ',' + str(id1) + ',' + str(
            id2) + ',' + str(weight) + '\n'

    with open(os.path.join(base_path, 'co_network.txt'), 'w', encoding='utf8') as f:
        f.write(line)
    logger.info("done!!!")
    return edge_dict


class sn_model():
    def __init__(self,
                 processes=10,
                 top_k=15,
                 graph_path='../temp/network/co_network.txt',
                 input_word_id=None):
        self.graph_path = graph_path
        self.processes = processes
        self.top_k = top_k
        self.network = read_network(graph_path)
        self.input_word_id = input_word_id
        self.num_of_nodes = self.network.number_of_nodes()
        self.num_of_edges = self.network.number_of_edges()

        logger.info('loaded network file ，totally {a} nodes ，{b} edges'.format(
            a=self.num_of_nodes, b=self.num_of_edges))

    def cal_sim(self, node, G):
        nodes = G.nodes
        if node not in nodes: return
        node_set = set(G.neighbors(node))
        n1_nodes_num = len(node_set)
        sim_dict = {}
        for n in nodes:
            neibor_set = set(G.neighbors(n))
            n2_nodes_num = len(neibor_set) + 1
            inter = node_set & neibor_set
            union = node_set | neibor_set

            coef1 = n2_nodes_num * 1.0 / (n2_nodes_num - len(inter) + 1)
            coef2 = n1_nodes_num * 1.0 / (n1_nodes_num - len(inter) + 1)
            jaccord = len(inter) * 1.0 / len(union) * coef1 * coef2
            sim_dict[n] = jaccord

        sorted_dict = sorted(sim_dict.items(),
                             key=lambda e: e[1],
                             reverse=True)[0:self.top_k]
        sorted_dict = [(G.nodes[k]['name'], v) for k, v in sorted_dict
                       if v > 0.25]
        return sorted_dict

    def synonym(self, input_word_id, network, lock, input_word_code_dict,
                id2word):
        line = ''
        cnt = 0
        for node in input_word_id:
            cnt += 1
            if str(node) not in network.nodes:
                continue
            node_name = id2word[node]
            node_name2 = network.nodes[str(node)]['name']
            if node_name != node_name2: continue
            node_code = input_word_code_dict[node_name]
            logger.info('process id = {b}, handling the {a} input word'.format(
                a=cnt, b=os.getpid()))
            synonym_dict = self.cal_sim(str(node), network)
            if synonym_dict is not None:
                temp_list = [k for (k, v) in synonym_dict]
                line += str(node_code) + '\t' + node_name + '\t' + '|'.join(
                    temp_list) + '\n'
        logger.info(
            'process id = {a}, start write file......'.format(a=os.getpid()))
        with lock:
            with open(os.path.join(os.path.dirname(self.graph_path), 'semantic_network_model_synonym.txt'),
                      'a',
                      encoding='utf8') as f:
                f.write(line)

    def synonym_detect(self, input_word_code_dict, id2word):
        import math
        lock = Lock()
        logger.info(' start detect synonym......')
        partition = math.ceil(len(self.input_word_id) / self.processes)
        start, end = 0, partition
        pro_list = []
        word_num = len(self.input_word_id)
        if word_num < self.processes:
            logger.info(
                'error!! the number of process is more than the number of input words'
            )
            return
        for i in range(self.processes):
            if end > word_num: break
            word_id = self.input_word_id[start:end]
            p = Process(target=self.synonym,
                        args=(word_id, self.network, lock,
                              input_word_code_dict, id2word))
            pro_list.append(p)
            p.start()
            start, end = end, min(end + partition, word_num)
        for p in pro_list:
            p.join()
        logger.info('done!!!')


def read_network(graph_file):
    G = nx.Graph()
    with open(graph_file, encoding='utf8') as f:
        line = f.readline()
        while line:
            row = line.split(",")
            if len(row) < 5:
                line = f.readline()
                continue
            node1, node2, id1, id2, weight = row[0], row[1], row[2], row[
                3], float(row[4])
            G.add_weighted_edges_from([(id1, id2, weight)])
            G.nodes[id1]['name'] = node1
            G.nodes[id2]['name'] = node2
            line = f.readline()
    return G


def synonym_detect(data, input_word_id, input_word_code_dict, id2word,
                   word2id, win_len, top_k, process_number, base_path):

    graph_file = os.path.join(base_path, 'co_network.txt')

    if os.path.exists(graph_file):
        os.remove(graph_file)
    out_path = os.path.join(base_path, 'semantic_network_model_synonym.txt')
    if os.path.exists(out_path):
        os.remove(out_path)

    args = (data, word2id, win_len, base_path)
    generate_network_from_corpus(args)

    model = sn_model(input_word_id=input_word_id,
                     processes=process_number,
                     graph_path=os.path.join(base_path, 'co_network.txt'),
                     top_k=top_k)
    model.synonym_detect(input_word_code_dict, id2word)


if __name__ == '__main__':
    split_file(file_line_size=100)