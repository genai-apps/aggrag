import io
import os
import shutil
import logging
import zipfile

import aiofiles

logger = logging.getLogger(__name__)


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


def zip_directory(folder_path):
    if not os.path.exists(folder_path):
        return None
    # Create an in-memory bytes buffer
    memory_file = io.BytesIO()

    # Create a zip file in the memory buffer
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname)

    # Seek to the start of the BytesIO object to allow Flask to send it
    memory_file.seek(0)

    return memory_file


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
