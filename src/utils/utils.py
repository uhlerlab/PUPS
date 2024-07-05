import os
from functools import reduce

FOLDER_1 = "/path/to/folder1"
FOLDER_2 = "/path/to/folder2"
DATA_FOLDERS = [
    FOLDER_1,
    FOLDER_2,
]


def get_data_path(file_name, preference = None):
    if preference is None: 
        selected_data_folder = None
        for data_folder in DATA_FOLDERS:
            if os.path.isdir(data_folder):
                selected_data_folder = data_folder
                break
        assert (
            selected_data_folder != None
        ), "This platform doesnt have a root datafolder defined"
    else:
        selected_data_folder = preference

    file_name = reduce(
        lambda file_name, prefix: file_name.replace(prefix, ""),
        [file_name] + DATA_FOLDERS,
    )
    return os.path.join(selected_data_folder, file_name)
