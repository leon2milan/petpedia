from pymongo import MongoClient
from pymongo.mongo_client import _MongoClientErrorHandler
from core.tools import Singleton, setup_logger
from config import get_cfg

logger = setup_logger()

@Singleton
class Mongo():
    def __init__(self, cfg, db=None) -> None:
        self.cfg = cfg
        self.conn = MongoClient(
            f'mongodb://{self.cfg.MONGO.USER}:{self.cfg.MONGO.PWD}@{self.cfg.MONGO.HOST}:{self.cfg.MONGO.PORT}/{db}'
        )
        if db is not None:
            self.db = self.conn[db]

    def get_state(self):
        return self.conn is not None and self.db is not None

    def insert_one(self, collection, data):
        if self.get_state():
            ret = self.db[collection].insert_one(data)
            return ret.inserted_id
        else:
            logger.error("mongo service is down. Please check.")
            raise _MongoClientErrorHandler

    def insert_many(self, collection, data):
        if self.get_state():
            ret = self.db[collection].insert_many(data)
            return ret
        else:
            logger.error("mongo service is down. Please check.")
            raise _MongoClientErrorHandler

    def update(self, collection, data):
        data_filter = {}
        data_revised = {}
        for key in data.keys():
            data_filter[key] = data[key][0]
            data_revised[key] = data[key][1]
        if self.get_state():
            return self.db[collection].update_many(data_filter, {
                "$set": data_revised
            }).modified_count
        else:
            logger.error("mongo service is down. Please check.")
            raise _MongoClientErrorHandler

    def find(self, collection, condition, columns=None):
        if self.get_state():
            if columns is None:
                return self.db[collection].find(condition)
            else:
                return self.db[collection].find(condition, columns)
        else:
            logger.error("mongo service is down. Please check.")
            raise _MongoClientErrorHandler

    def delete(self, collection, condition):
        if self.get_state():
            return self.db[collection].delete_many(
                filter=condition).deleted_count
        else:
            logger.error("mongo service is down. Please check.")
            raise _MongoClientErrorHandler

    def clean(self, collection):
        if self.get_state():
            return self.db[collection].remove()
        else:
            logger.error("mongo service is down. Please check.")
            raise _MongoClientErrorHandler


if __name__ == "__main__":
    cfg = get_cfg()
    mongo = Mongo(cfg)