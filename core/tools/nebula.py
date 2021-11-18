from core.tools import Singleton
from config import get_cfg


from nebula2.gclient.net import ConnectionPool
from nebula2.Config import Config

@Singleton
class Nebula():
    def __init__(self, cfg):
        self.cfg = cfg
        self.session = self.get_session()

    def get_session(self):
        config = Config()
        config.max_connection_pool_size = 10
        # 初始化连接池
        connection_pool = ConnectionPool()
        # 如果给定的服务器正常，则返回true，否则返回false。
        ok = connection_pool.init([(self.cfg.NEBULA.HOST, self.cfg.NEBULA.PORT)], config)

        # 方法1：控制连接自行释放。
        # 从连接池中获取会话
        session = connection_pool.get_session(self.cfg.NEBULA.USER, self.cfg.NEBULA.PWD)
        return session
    
    def query(self, nsql):
        try:
            resp = self.session.execute(nsql)
            assert resp.is_succeeded(), resp.error_msg()
        except Exception as x:
            import traceback
            return resp.is_succeeded(), traceback.format_exc()
        return resp.is_succeeded(), resp


if __name__ == '__main__':
    cfg = get_cfg()
    nebula = Nebula(cfg)
