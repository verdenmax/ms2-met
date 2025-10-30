
import os


def get_filename_stem(filepath: str) -> str:
    """ 从路径中获取这个文件的文件名，去除扩展 """
    filename = os.path.basename(filepath)
    stem, _ = os.path.splitext(filename)
    return stem
