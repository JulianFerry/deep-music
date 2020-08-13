# -*- coding: utf-8 -*-

import subprocess
from distutils.command.build import build as _build  # type: ignore
import setuptools


# This class handles the pip install mechanism.
class build(_build):  # pylint: disable=invalid-name
    """A build command class that will be invoked during package install.
    The package built using the current setup.py will be staged and later
    installed in the worker using `pip install package'. This class will be
    instantiated during install for this specific scenario and will trigger
    running the custom commands specified.
    """
    sub_commands = _build.sub_commands + [('CustomCommands', None)]


CUSTOM_COMMANDS = [
    ['apt-get', 'update'],
    ['apt-get', '--assume-yes', 'install', 'libsndfile1']]


class CustomCommands(setuptools.Command):
    """A setuptools Command class able to run arbitrary commands."""
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def RunCustomCommand(self, command_list):
        print('Running command: %s' % command_list)
        p = subprocess.Popen(
                command_list,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
        stdout_data, _ = p.communicate()
        print('Command output: %s' % stdout_data)
        if p.returncode != 0:
            raise RuntimeError(
                    'Command %s failed: exit code: %s' % (command_list, p.returncode))

    def run(self):
        for command in CUSTOM_COMMANDS:
            self.RunCustomCommand(command)


setuptools.setup(
    long_description='',
    name='preprocessing',
    version='0.1.0',
    description='Audio preprocessing for deep-music project',
    python_requires='==3.*,>=3.7.0',
    author='JulianFerry',
    author_email='julianferry94@gmail.com',
    packages=['preprocessing'],
    install_requires=[
        'apache-beam[gcp]==2.*,>=2.23.0',
        'numba==0.48.0',
        'requests==2.*,>=2.24.0',
        'fsspec==0.8.0',
        'gcsfs==0.6.2'
    ],
    cmdclass={
        # Command class instantiated and run during pip install scenarios.
        'build': build,
        'CustomCommands': CustomCommands,
    }
)
