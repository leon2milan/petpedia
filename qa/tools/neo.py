from pymongo import MongoClient
from qa.tools import Singleton, setup_logger
from config import get_cfg
from neo4j import GraphDatabase

logger = setup_logger(name='mongo')


@Singleton
class NEO4J:
    def __init__(self, cfg):
        self.cfg = cfg
        self.driver = GraphDatabase.driver(
            f'bolt://{self.cfg.NEO4J.HOST}:{self.cfg.NEO4J.PORT}',
            auth=(self.cfg.NEO4J.USER, self.cfg.NEO4J.PWD))
        self.db = self.driver.session()

    def __del__(self):
        self.close()

    def close(self):
        self.driver.close()

    def run(self, cql):
        result = self.db.read_transaction(lambda tx: list(tx.run(cql)))
        return result


if __name__ == "__main__":
    cfg = get_cfg()
    neo4j = NEO4J(cfg)
    cypher_sql = "match(n)-[r]->(m) where n.name ='{}' return type(r), keys(n)".format(
        '阿尔卑斯达切斯勃拉克犬')
    import time
    s = time.time()
    print(neo4j.run(cypher_sql), time.time() - s)
