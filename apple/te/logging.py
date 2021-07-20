logger = logging.getLogger('te') 
logger.debug("debug log test")

from os.path import join

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'fileFormat': {
            'format': '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
            'datefmt': '%d/%b/%Y %H:%M:%S'
        }
    },
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': join(BASE_DIR, 'logs/logfile.log'),
            'formatter': 'fileFormat'
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
        },

    },
    'loggers': {
        'post': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
        },
        'django.db.backends': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
        },
    }
}