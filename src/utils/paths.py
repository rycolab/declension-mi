import os
from .utils import mkdir


class ApplicationPaths:

    @staticmethod
    def application_root():
        return os.path.abspath(os.getcwd())

    @staticmethod
    def get_path(default_path, subfolder_path=''):
        path = ApplicationPaths.application_root() + os.sep + default_path + subfolder_path + os.sep
        mkdir(path)
        return path

    @staticmethod
    def config(file_name='', subfolder_path=''):
        return ApplicationPaths.get_path("config/", subfolder_path) + file_name

    @staticmethod
    def logs(file_name='', subfolder_path=''):
        return ApplicationPaths.get_path("logs/", subfolder_path) + file_name

    @staticmethod
    def datasets(file_name='', subfolder_path=''):
        return ApplicationPaths.get_path("datasets/", subfolder_path) + file_name

    @staticmethod
    def assets(file_name='', subfolder_path=''):
        return ApplicationPaths.get_path("assets/", subfolder_path) + file_name
