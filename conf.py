from config import get_cfg, ROOT

cfg = get_cfg()

# 项目目录地址
project_dir = ROOT

# 把gunicorn工作目录切换到项目根路径
chdir = project_dir

# 并行进程数
workers = cfg.WEB.WORKER

# 指定每个进程开启的线程数
threads = cfg.WEB.THREADS

timeout=120

# 监听端口
bind = f'{cfg.WEB.HOST}:{cfg.WEB.PORT}'

# 是否设为守护进程
daemon = 'true'

# 工作模式：协程
# 默认为sync异步，类型：sync, eventlet, gevent, tornado, gthread, gaiohttp
# gevent 更快，需要安装 gevent模块
worker_class = cfg.WEB.WORK_CLASS

# 客户端最大并发量，默认为1000
# worker_connections = 512
# 等待连接的最大数，默认2048
# backlog = 1024

# # 进程文件
pidfile = cfg.WEB.PID_FILE

# 访问日志和错误日志
accesslog = cfg.WEB.LOG
errorlog = cfg.WEB.ERROR_LOG

# 日志级别
loglevel = cfg.WEB.LOG_LEVEL
