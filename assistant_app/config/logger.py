import logging


def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logging.getLogger("telegram").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    return logging.getLogger(__name__)


logger = setup_logging(debug=False)
