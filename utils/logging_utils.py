import logging
from logging import INFO, DEBUG, ERROR, WARNING, CRITICAL
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
import os
import sys
import datetime
import gzip
import shutil
from enum import Enum
import re


class RotationType(Enum):
    TIMED = 1
    SIZED = 2


class RotationIntervalUnit(Enum):
    SECONDS = "s"
    MINUTES = "m"
    HOURS = "h"
    WEEKS = "w"
    MIDNIGHT = "midnight"


class ColouredLogger(logging.Logger):
    _level_color = {
        10: 36,
        20: 34,
        30: 33,
        40: 31,
        50: 31,
    }

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.mode = "stream"

    def _rotator(self, source, dest):
        with open(source, "rb") as f_in:
            with gzip.open(f"{dest}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(source)

    def _timed_filer(self, default_name):
        now = datetime.datetime.now()
        folder_name = f'{os.path.dirname(self.logging_path)}/{now.strftime("%Y")}/{now.strftime("%Y-%m")}'
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
        base_name: str = os.path.basename(default_name)
        return f"{folder_name}/{base_name}"

    def _sized_filer(self, default_name: str):
        now = datetime.datetime.now()
        folder_name = f'{os.path.dirname(self.logging_path)}/{now.strftime("%Y")}/{now.strftime("%Y-%m")}'
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
        base_name: str = os.path.basename(default_name)
        base_name, counter = base_name.rsplit(".", 1)
        counter = len(os.listdir(folder_name))
        return f"{folder_name}/{base_name}.{int(counter) + 1}"

    def _expression_to_bytes(self, expression: str):
        found = re.match(r"^(\d+)(?=M|G|$)(M|G|)$", expression)
        if found is None:
            raise ValueError("The provided log file size expression is invalid.")
        quantity = int(found.group(1))
        multiplier = 1
        if found.group(2) == "M":
            multiplier = 1000
        elif found.group(2) == "G":
            multiplier = 1000 * 1000
        return quantity * multiplier

    def auto_configure(
        self,
        debug: bool = False,
        logging_path: str = "./logs.log",
        rotate_logs=False,
        *,
        rotation_type=RotationType.TIMED.name.lower(),
        rotation_interval_unit=RotationIntervalUnit.MIDNIGHT.value,
        rotation_interval=5,
        log_max_size="100M",
    ):
        logging_level = logging.INFO if not debug else logging.DEBUG
        logging_formatter = logging.Formatter(
            fmt="%(asctime)s %(threadName)s %(name)s %(module)s #%(lineno)d %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if os.getpgrp() == os.tcgetpgrp(sys.stdout.fileno()):
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging_formatter)
            stream_handler.setLevel(logging_level)
            logging.getLogger().addHandler(stream_handler)
            logging.getLogger().mode = "stream"
        else:
            if not os.path.isdir(os.path.dirname(logging_path)):
                os.makedirs(os.path.dirname(logging_path))
            self.logging_path = logging_path
            file_handler = logging.FileHandler(filename=logging_path)
            if rotate_logs:
                if rotation_type == RotationType.TIMED.name.lower():
                    file_handler = TimedRotatingFileHandler(
                        filename=logging_path,
                        when=rotation_interval_unit,
                        interval=rotation_interval,
                    )
                    file_handler.namer = self._timed_filer
                else:
                    file_handler = RotatingFileHandler(
                        filename=logging_path,
                        maxBytes=self._expression_to_bytes(log_max_size),
                        backupCount=1000,
                    )
                    file_handler.namer = self._sized_filer
                file_handler.rotator = self._rotator
            file_handler.setFormatter(logging_formatter)
            file_handler.setLevel(logging_level)
            logging.getLogger().addHandler(file_handler)
            logging.getLogger().mode = "file"

    def _log_with_color(self, level: int, message: str):
        if logging.getLogger().mode is None or logging.getLogger().mode == "file":
            return message
        return f"\033[{self._level_color[level]}m{message}\033[0m"

    def info(self, msg, *args, **kwargs):
        if "stack_level" in kwargs:
            stack_level = kwargs.pop("stack_level")
            kwargs.pop("stack_level")
        else:
            stack_level = 2
        self._log(
            INFO,
            self._log_with_color(INFO, msg),
            args,
            stacklevel=stack_level,
            **kwargs,
        )

    def debug(self, msg, *args, **kwargs):
        if "stack_level" in kwargs:
            stack_level = kwargs.pop("stack_level")
        else:
            stack_level = 2
        self._log(
            DEBUG,
            self._log_with_color(DEBUG, msg),
            args,
            stacklevel=stack_level,
            **kwargs,
        )

    def warning(self, msg, *args, **kwargs):
        if "stack_level" in kwargs:
            stack_level = kwargs.pop("stack_level")
        else:
            stack_level = 2
        self._log(
            WARNING,
            self._log_with_color(WARNING, msg),
            args,
            stacklevel=stack_level,
            **kwargs,
        )

    def error(self, msg, *args, **kwargs):
        if "stack_level" in kwargs:
            stack_level = kwargs.pop("stack_level")
        else:
            stack_level = 2
        self._log(
            ERROR,
            self._log_with_color(ERROR, msg),
            args,
            stacklevel=stack_level,
            **kwargs,
        )

    def critical(self, msg, *args, **kwargs):
        if "stack_level" in kwargs:
            stack_level = kwargs.pop("stack_level")
        else:
            stack_level = 2
        self._log(
            CRITICAL,
            self._log_with_color(CRITICAL, msg),
            args,
            stacklevel=stack_level,
            **kwargs,
        )


logging.setLoggerClass(ColouredLogger)
