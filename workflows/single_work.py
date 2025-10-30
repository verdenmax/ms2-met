
import logging

from spectrum.psm_info import PSMInfo
from spectrum.dia_data import DIAData

from configparser import ConfigParser


def single_pair_work(
    psm: PSMInfo,
    dia_data: DIAData,
    config: ConfigParser
):

    # TODO:

    logging.info(f"处理信息 {psm}")
