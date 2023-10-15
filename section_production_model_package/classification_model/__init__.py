import logging

from pathlib import Path
import sys

sys.path.append(
    str(Path().absolute().parent))  # this command beginning in the folder of file prepare_fastapi_api.py everywhere you launch it, this path is in the folder section-production-model-package

from classification_model.configuration import core

# It is strongly advised that you do not add any handlers other than
# NullHandler to your library’s loggers. This is because the configuration
# of handlers is the prerogative of the application developer who uses your
# library. The application developer knows their target audience and what
# handlers are most appropriate for their application: if you add handlers
# ‘under the hood’, you might well interfere with their ability to carry out
# unit tests and deliver logs which suit their requirements.
# https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
logging.getLogger(core.config.app_config.package_name).addHandler(logging.NullHandler())


with open(core.PACKAGE_ROOT / "classification_model" / "VERSION") as version_file:
    __version__ = version_file.read().strip()
