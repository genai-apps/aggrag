import os
import shutil
import logging

import aiofiles

logger = logging.getLogger(__name__)


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


def get_time_taken(start, interim, final):
    """
    Calculate and return the timing statistics for an operation given the start,
    interim, and final times.

    Args:
        start (float): The start time of the operation.
        interim (float): The interim checkpoint time of the operation.
        final (float): The end time of the operation.

    Returns:
        dict: A dictionary containing 'response', 'eval', and 'total' timing values.
    """
    eval_time = round(final - interim, 2)
    response_time = round(interim - start, 2)
    return {'total': round(response_time, 2)}
    # return {'response': response_time, 'eval': eval_time, 'total': round(eval_time + response_time, 2)}
