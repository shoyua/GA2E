import logging
import os
import sys


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL}

    def __init__(self, filename, data='data', level='info'):
        self.logger = logging.getLogger("all in one by gan and vae ")

        # create file handler
        log_path = os.path.join(sys.path[0], 'logs_paper', data)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_path = os.path.join(log_path, filename)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        # create formatter
        fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(message)s"
        datefmt = "%a %d %b %Y %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt)
        fh.setFormatter(formatter)  # 设置写入文件格式
        self.logger.addHandler(fh)
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕输出
        sh.setFormatter(formatter)  # 设置屏幕显示格式
        self.logger.addHandler(sh)  # 把对象加到logger里


if __name__ == '__main__':
    log = Logger('test.log', level='info')
    log.logger.info('1\t2\t3')
