import argparse


def main():
    # 设置程序参数， --configpath 来设置配置文件路径
    parser = argparse.ArgumentParser(description='利用代谢标记发展MS2检验技术')
    parser.add_argument(
        '--configpath', help='config.ini 文件路径，默认为 ./config.ini',
        default="./config.ini")
    args = parser.parse_args()


if __name__ == "__main__":
    main()
