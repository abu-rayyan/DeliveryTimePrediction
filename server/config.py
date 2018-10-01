class BaseConfig(object):
    """Base Configurations"""
    SECRET_KEY = 'secret'
    DEBUG = True
    TESTING = False
    # more config vars ca be added here


class ProductionConfig(BaseConfig):
    """Production Specific Configurations"""
    DEBUG = False
    SECRET_KEY = 'secret'  # TODO: supply a secret key file here


class StagingConfig(BaseConfig):
    """Staging Specific Configurations"""
    DEBUG = True


class DevelopmentConfig(BaseConfig):
    """Development Specific Configurations"""
    DEBUG = True
    TESTING = True
    SECRET_KEY = 'secret'
