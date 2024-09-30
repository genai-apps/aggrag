import logging
import os
from logging.handlers import RotatingFileHandler
from library.aggrag.core.config import settings

# Create the logs directory if it doesn't exist
log_dir = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up the logger
logger = logging.getLogger('app_logger')
logger.setLevel(settings.LOGGING_LEVEL)

# Create a file handler for writing logs to a file
file_handler = RotatingFileHandler(os.path.join(log_dir, 'app.log'), maxBytes=10 * 1024 * 1024, backupCount=5)
file_handler.setLevel(settings.LOGGING_LEVEL)

# Create a formatter for the log messages
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

# Set the logger as the global app_logger
app_logger = logger