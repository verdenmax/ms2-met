import argparse
import configparser
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
    config.read(args.configpath)

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
