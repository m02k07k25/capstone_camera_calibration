"""Shared defaults and command-line value parsers."""

from __future__ import annotations

import argparse


PROJECT_NAME = "capstone_camera_calibration"


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


DEFAULT_COLS = 7


DEFAULT_ROWS = 5


DEFAULT_SQUARE_SIZE = 35.0


DEFAULT_MARKER_SIZE = 24.5


DEFAULT_DICTIONARY = "DICT_5X5_100"


DEFAULT_PIXELS_PER_SQUARE = 400


MIN_CHARUCO_CORNERS = 8


DEFAULT_CAPTURE_WIDTH = 1920


DEFAULT_CAPTURE_HEIGHT = 1080


DEFAULT_PREVIEW_WIDTH = 1600


DEFAULT_PREVIEW_HEIGHT = 900


DEFAULT_CAPTURE_INTERVAL = 4.0


DEFAULT_STABLE_SECONDS = 2.0


DEFAULT_STABILITY_PIXELS = 5.0


BACKEND_CHOICES = ("auto", "dshow", "msmf")


ARUCO_DICTIONARIES = (
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_6X6_50",
    "DICT_6X6_100",
)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 정수를 입력하세요.")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("0 이상인 정수를 입력하세요.")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 숫자를 입력하세요.")
    return number
