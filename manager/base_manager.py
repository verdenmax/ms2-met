import logging
import os
import pickle

import manager


class BaseManager:
    def __init__(
        self,
        path: None | str = None,
        load_from_file: bool = True,
        figure_path: None | str = None,
    ):
        """
        BaseManager：
        用来从文件中管理数据
        保存三个变量，文件的路径、是否加载文件、figure 路径
        """
        self._path = path
        self.is_loaded_from_file = False
        self.figure_path = figure_path
        self._version = manager.__version__

        if load_from_file:
            self.load()

    @property
    def path(self):
        """ pickle 的文件路径 """
        return self._path

    @property
    def is_loaded_from_file(self):
        """ 检查是否加载 """
        return self._is_loaded_from_file

    @is_loaded_from_file.setter
    def is_loaded_from_file(self, value):
        """ 是否加载文件 """
        self._is_loaded_from_file = value

    def save(self):
        """保存到 pickle."""
        if self.path is None:
            return

        try:
            # 写入的时候，先写入一个临时文件，最后再替换
            # 这样保证了文件是对的，不会中间损坏
            temp_path = self.path + ".tmp"

            with open(temp_path, "wb") as f:
                pickle.dump(self, f)
                f.flush()  # 确保数据写入磁盘

            # 原子性替换：重命名是原子操作
            os.replace(temp_path, self.path)
        except Exception as e:

            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

            logging.error(
                f"Failed to save {self.__class__.__name__} to {
                    self.path}: {str(e)}")

    def load(self):
        """从 pickle 中加载."""
        if self.path is None:
            logging.info(f"无 {self.__class__.__name__} pickle 文件，将会初始化")
            return
        elif not os.path.exists(self.path):
            logging.info(f"无 {self.__class__.__name__} pickle 文件，将会初始化")
            return

        try:
            with open(self.path, "rb") as f:
                loaded_state = pickle.load(f)

                if loaded_state._version == self._version:
                    self.__dict__.update(loaded_state.__dict__)
                    self.is_loaded_from_file = True
                    logging.info(
                        f"Loaded {self.__class__.__name__} from {self.path}"
                    )
                else:
                    logging.warn("错误的版本")
        except Exception:
            logging.error("加载错误")
