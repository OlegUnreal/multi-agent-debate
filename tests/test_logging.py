import json
import logging

from debate.logging_config import JsonFormatter, setup_logging, get_logger


def test_json_formatter_emits_object():
    fmt = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello %s", args=("world",), exc_info=None,
    )
    out = fmt.format(record)
    data = json.loads(out)
    assert data["level"] == "INFO"
    assert data["msg"] == "hello world"
    assert "ts" in data


def test_setup_logging_is_idempotent():
    setup_logging()
    setup_logging()
    log = get_logger("debate.test")
    log.info("ping")
