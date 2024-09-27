import logging
import os
from library.aggrag.core.config import settings


class Logger:

    formatter: logging.Formatter

    def __init__(self):
        log_format: str = '%(asctime)s %(name)s %(levelname)s | %(message)s'
        self.formatter = logging.Formatter(log_format)

    # Create a filter to only process given level records
    class LevelFilter(logging.Filter):

        def __init__(self, log_lvl: int):
            super().__init__()
            self.log_lvl = log_lvl

        def filter(self, record):
            return record.levelno == self.log_lvl

    def create_logger(
            self,
            log_lvl: int,
            file_name: str,
            filter_higher_lvl: bool = True,
            debug: bool = False 
    ):
        log_handler = logging.FileHandler(f'logs/{file_name}.log')
        log_handler.setLevel(log_lvl)
        log_handler.setFormatter(self.formatter)
        if debug and log_lvl == logging.DEBUG:  
            log_handler.setLevel(logging.DEBUG)

        if filter_higher_lvl:
            log_handler.addFilter(self.LevelFilter(log_lvl))

        return log_handler

    def configure_logs(self):
        # setup loggers

        if not os.path.exists('logs'):
            os.makedirs('logs')



        logging.basicConfig(
            level=settings.LOGGING_LEVEL
        )

        info_handler = self.create_logger(log_lvl=logging.INFO, file_name='info', debug=False)
        warning_handler = self.create_logger(log_lvl=logging.WARNING, file_name='warning', debug=False)
        debug_handler = self.create_logger(log_lvl=logging.DEBUG, file_name='debug', debug=True)

        error_handler = self.create_logger(log_lvl=logging.ERROR, file_name='error', filter_higher_lvl=False, debug=False)

        logger = logging.getLogger()

        logger.addHandler(info_handler)
        logger.addHandler(warning_handler)
        logger.addHandler(debug_handler)  
        logger.addHandler(error_handler)


app_logger = Logger()
