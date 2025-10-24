import argparse
import configparser
from banner import banner

import logging
from rich.logging import RichHandler


def main():
    # 设置程序参数， --configpath 来设置配置文件路径
    parser = argparse.ArgumentParser(description='利用代谢标记发展MS2检验技术')
    parser.add_argument(
        '--configpath', help='config.ini 文件路径，默认为 ./config.ini',
        default="./config.ini")
    args = parser.parse_args()

    # 解析配置文件
    config = configparser.ConfigParser()
    config.read(args.configpath)

    # 注册日志
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

    # 展示开始的banner
    banner.show_start_banner()

    # 展示程序运行结束banner
    banner.show_end_banner()


if __name__ == "__main__":
    main()
