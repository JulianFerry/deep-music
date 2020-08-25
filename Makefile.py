from easymake import target
from easymake.helpers import Globals, Shell

g = Globals.from_path('project/Makefile')
shell = Shell(g)


@target
def download(*args, **kwargs):
    """Run easymake on download package"""
    shell.ezmake(args, kwargs, cwd='${project_path}/src/download')


@target
def preprocessing(*args, **kwargs):
    """Run easymake on preprocessing package"""
    shell.ezmake(args, kwargs, cwd='${project_path}/src/preprocessing')


@target
def trainer(*args, **kwargs):
    """Run easymake on trainer package"""
    shell.ezmake(args, kwargs, cwd='${project_path}/src/trainer')


@target
def evaluate(*args, **kwargs):
    """Run easymake on evaluate package"""
    shell.ezmake(args, kwargs, cwd='${project_path}/src/evaluate')
