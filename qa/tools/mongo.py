from pymongo import MongoClient
from pymongo.mongo_client import _MongoClientErrorHandler
from qa.tools import Singleton, setup_logger
from config import get_cfg

logger = setup_logger(name='mongo')
__all__ = ['Mongo']


@Singleton
class Mongo():
    __slot__ = ['cfg', 'conn', 'db']

    def __init__(self, cfg, db=None) -> None:
        self.cfg = cfg
        logger.info(f"connect to {self.cfg.BASE.ENVIRONMENT}")
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
            return self.db.drop_collection(collection)
        else:
            logger.error("mongo service is down. Please check.")
            raise _MongoClientErrorHandler

    def show_dbs(self):
        dbs = self.conn.database_names()   # pymongo获取mongodb实例下所有数据库名称
        return dbs
    
    def show_collections(self):
        result = self.db.list_collection_names(session=None)    #  pymongo获取指定数据库的集合名称
        return result

if __name__ == "__main__":
    cfg = get_cfg()
    mongo = Mongo(cfg, 'qa')
    print(mongo.show_dbs())
    print(mongo.show_collections())