import time

import numpy as np
from scipy import spatial
import tritonclient.http
from config import get_cfg
from transformers import AutoTokenizer

from utils import print_timings, setup_logging, track_infer_time

setup_logging()
tokenizer = AutoTokenizer.from_pretrained(
    "/workspaces/ai-petpedia/models/pretrained_model/macbert_base")

text = "狗狗偶尔尿血怎么办"

test = [['哈士奇拆家怎么办', '猫拉肚子不吃东西'], ['哈士奇拆家怎么办', '哈士奇总是乱咬东西怎么办'],
        ['哈士奇拆家怎么办', '狗拆家怎么办'], ['猫癣', '什么是猫癣'], ['什么是猫癣', '得了猫癣怎么办'],
        ['猫癣', '猫癣转阴']]


class TritonServer:
    def __init__(self, cfg, model_name) -> None:
        self.cfg = cfg
        self.url = f'{self.cfg.TRITON.HOST}:{self.cfg.TRITON.PORT}'
        self.triton_client = tritonclient.http.InferenceServerClient(
            url=self.url, verbose=False)
        self.model_name = model_name
        self.model_version = "1"
        assert self.triton_client.is_model_ready(
            model_name=self.model_name, model_version=self.model_version
        ), f"model {self.model_name} not yet ready"

    def get_embeding(self, text):
        batch_size = 1

        query = tritonclient.http.InferInput(name="TEXT",
                                             shape=(batch_size, ),
                                             datatype="BYTES")
        model_score = tritonclient.http.InferRequestedOutput(name="embedding",
                                                             binary_data=False)

        query.set_data_from_numpy(np.asarray([text] * batch_size,
                                             dtype=object))
        x = self.triton_client.infer(model_name=self.model_name,
                                     model_version=self.model_version,
                                     inputs=[query],
                                     outputs=[model_score])
        return x.as_numpy("embedding")

    def similarity(self, str1, str2):
        str1 = self.get_embeding(str1)
        str2 = self.get_embeding(str2)
        return 1 - spatial.distance.cosine(str1, str2)


if __name__ == '__main__':
    cfg = get_cfg()
    triton = TritonServer(cfg, 'simcse')
    for i in test:
        s = time.time()
        print(
            f"query: {i[0]}, candidate: {i[1]}, score: {triton.similarity(i[0], i[1])}, time: {time.time() - s}"
        )
