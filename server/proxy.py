import logging
import os

logger = logging.getLogger(__name__)

proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]


def setup_proxy():
    logger.info("[Proxy] Setting up HTTP/HTTPS proxy for all requests")
    _PROXY = os.getenv("PROXY_URL", "http://127.0.0.1:28083")
    for _key in proxy_keys:
        os.environ.setdefault(_key, _PROXY)
    logger.info("[Proxy] HTTP/HTTPS 代理已设置: %s", _PROXY)


def clear_proxy():
    logger.info("[Proxy] Clearing HTTP/HTTPS proxy settings")
    for _key in proxy_keys:
        os.environ.pop(_key, None)
    logger.info("[Proxy] HTTP/HTTPS 代理已清除")