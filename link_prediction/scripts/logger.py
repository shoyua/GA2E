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
        self.logger = logging.getLogger("ga2e")
        # create file handler
        log_path = os.path.join(sys.path[0], 'logs_paper_1128', data)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_path = os.path.join(log_path, filename)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        # create formatter
        fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(message)s"
        datefmt = "%a %d %b %Y %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt)
        fh.setFormatter(formatter)  
        self.logger.addHandler(fh)
        self.logger.setLevel(self.level_relations.get(level))  
        sh = logging.StreamHandler()  
        sh.setFormatter(formatter)  
        self.logger.addHandler(sh)  
if __name__ == '__main__':
    log = Logger('test.log', level='info')
    log.logger.info('1\t2\t3')
