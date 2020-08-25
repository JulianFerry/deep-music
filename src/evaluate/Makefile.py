from easymake import target
from easymake.helpers import Globals, Shell

g = Globals.from_path('project/src/package/Makefile')
shell = Shell(g, cwd=g.package_path)


# `ezmake notebook`
@target
def notebook():
    """Run jupyter notebook server"""
    shell.run('poetry run jupyter notebook')


# `ezmake tensorboard`
@target
def tensorboard():
    """Run tensorboard server"""
    shell.run('poetry run tensorboard --logdir=${project_path}/trainer-output')
