"""mkdocs hook: 过滤 griffe 在 ``**kwargs`` / 旧式 docstring 上的噪音警告。

仅过滤"无注解的 ``**kwargs``"与个别已知遗留参数,真实问题(找不到引用、
断链等)依然会触发 strict 模式失败。
"""

from __future__ import annotations

import logging
import re

_NOISE_PATTERNS = [
    re.compile(r"No type or annotation for parameter '\*\*kwargs'"),
    re.compile(r"No type or annotation for parameter 'ocr_model'"),
    re.compile(r"No type or annotation for returned value 'Result'"),
    re.compile(r"No type or annotation for returned value 1"),
]


class _GriffeNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p.search(msg) for p in _NOISE_PATTERNS)


_FILTER = _GriffeNoiseFilter()


def _install_filter() -> None:
    # 装到所有现有 logger,以及未来才创建的 logger 的根
    for name in (None, "griffe", "mkdocs", "mkdocs.plugins.mkdocstrings"):
        logging.getLogger(name).addFilter(_FILTER)
    # 兜底:遍历当前已注册的所有 logger
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(logger, logging.Logger) and ("griffe" in name or "mkdocs" in name):
            logger.addFilter(_FILTER)


# 模块加载即注册(mkdocs 在解析 hooks: 时 import 本文件)
_install_filter()


def on_startup(*_, **__) -> None:
    _install_filter()


def on_config(config, **__):
    _install_filter()
    return config
