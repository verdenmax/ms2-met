import logging


def show_start_banner():
    logging.info("开始运行")

    try:
        with open("./banner/welcome.txt", 'r', encoding='utf-8') as banner:
            banner_content = banner.read()

            logging.info(banner_content)
    except Exception:
        text_info = """
                \033[38;5;196m╔══════════════════════════════╗\033[0m
                \033[38;5;214m║          ms2  check          ║\033[0m
                \033[38;5;118m║          v 0.0.1             ║\033[0m
                \033[38;5;196m╚══════════════════════════════╝\033[0m
        """

        print(text_info)


def show_end_banner():
    logging.info("finished")
