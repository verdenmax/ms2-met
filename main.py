import argparse
import configparser
import os
from banner import banner

import logging
from rich.logging import RichHandler


from workflows.pair_flow import PairFlow


def main():
    # 设置程序参数， --configpath 来设置配置文件路径
    parser = argparse.ArgumentParser(description='利用代谢标记发展MS2检验技术')
    parser.add_argument(
        '--configpath', help='config.ini 文件路径，默认为 ./config.ini',
        default="./config.ini")
    parser.add_argument(
        '--logpath', help='日志文件路径，默认为 ./ms2.log',
        default="./ms2.log")
    args = parser.parse_args()

    # 解析配置文件
    config = configparser.ConfigParser()
    read_files = config.read(args.configpath)
    if not read_files:
        raise FileNotFoundError(
            f"配置文件不存在或无法读取: '{args.configpath}' "
            f"(configparser.read() 返回空列表)")
    if not config.sections():
        raise ValueError(
            f"配置文件 '{args.configpath}' 内容为空或无 [section] "
            f"(读取成功但 sections=[])")

    # 确保日志文件父目录存在，避免 FileHandler 失败
    log_dir = os.path.dirname(args.logpath)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # 设置日志文件handle
    file_handler = logging.FileHandler(args.logpath, encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # 注册日志
    logging.basicConfig(level=logging.INFO, handlers=[
                        RichHandler(), file_handler])

    # 展示开始的banner
    banner.show_start_banner()

    # 进入 workflow , 系统的处理
    workflow = PairFlow(workname="main", config=config,
                        work_path="./workspace")

    # 运行
    workflow.run()

    # 展示程序运行结束banner
    banner.show_end_banner()


if __name__ == "__main__":
    main()
