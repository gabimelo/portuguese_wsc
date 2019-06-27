import logging

CUSTOM_FIELD = 'custom_field'


def handle_extra_params(logger_func):
    def __handle_extra_params_and_call(self, msg, extra={}):
        if CUSTOM_FIELD not in extra or extra[CUSTOM_FIELD] is None:
            extra[CUSTOM_FIELD] = ''
        elif extra[CUSTOM_FIELD] != '' and extra[CUSTOM_FIELD][-3:] != ' - ':
            extra[CUSTOM_FIELD] += ' - '
        return logger_func(self, msg, extra)

    return __handle_extra_params_and_call


class Logger(object):
    def __init__(self):
        # change to value of CUSTOM_FIELD in line below
        format = '%(levelname)s %(asctime)-15s: %(custom_field)s%(message)s'
        logging.basicConfig(format=format)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    @handle_extra_params
    def info(self, msg, extra):
        return self.logger.info(msg, extra=extra)

    @handle_extra_params
    def error(self, msg, extra):
        return self.logger.error(msg, extra=extra)

    @handle_extra_params
    def warning(self, msg, extra):
        return self.logger.warning(msg, extra=extra)
