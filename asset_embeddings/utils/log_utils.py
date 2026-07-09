"""In-class exception-logging decorator used across the pipeline.

``log_exceptions_inclass`` wraps a method so that any exception is logged to the instance's
logger (``self.logger`` by default) and to a timestamped file under ``logs/error/``, then
re-raised. Colored console formatting and logger construction live in
``asset_embeddings.logger`` and ``asset_embeddings.preparers.Logger_Preparer``.
"""

import os
import sys
import logging
from functools import wraps
from datetime import datetime


def log_exceptions_inclass(logger_attr: str = "logger"):
    """Decorator factory returning a decorator that catches and logs exceptions.

    The returned decorator:

    1. Logs the exception via the instance's logger attribute (``self.logger`` by default).
    2. Also writes the exception to a timestamped file under ``logs/error/`` using a dedicated ErrorLogger.

    Args:
        logger_attr: Name of the logger attribute on the instance. Defaults to ``"logger"``.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 获取主 logger（如 self.logger）
            main_logger = getattr(self, logger_attr, None)

            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                # 获取当前时间戳，用于日志文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                log_filename = f"{self.__class__.__name__}_{timestamp}.log"
                log_filepath = os.path.join("logs", "error", log_filename)

                # 确保目录存在
                os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

                # 获取或创建专用的 ErrorLogger
                error_logger = logging.getLogger("ErrorLogger")
                error_logger.setLevel(logging.ERROR)

                # 避免重复添加 handler
                if not error_logger.handlers:
                    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
                    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                    file_handler.setFormatter(file_formatter)
                    error_logger.addHandler(file_handler)

                # 记录异常到专用日志文件
                error_logger.error(f"Error occurred while executing command: {' '.join(sys.argv)}")
                error_logger.error(f"Exception in {func.__name__} of {self.__class__.__name__}", exc_info=True)

                # 同时在主 logger 中记录（如果存在）
                if main_logger is not None:
                    main_logger.error(f"Exception occurred in {func.__name__}", exc_info=True)
                else:
                    print(f"[Warning] {logger_attr} not found in {self.__class__.__name__}")

                # 重新抛出异常
                raise e

        return wrapper

    return decorator
