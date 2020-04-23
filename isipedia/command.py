'''Define markdown commands
'''
# from contextlib import contextmanager
commands_register = {}
contexts_register = []

# plumbing to register figures
figures_register = {}

def isipediafigure(name):
    """decorator to register the figure names and make them available in jinja2
    """
    def decorator(cls):
        figures_register[name] = cls
        return cls

    return decorator


def markdowncontext(f):
    contexts_register.append(f)
    return f

def markdowncommand(f):
    """decorator to register a markdown command and make it available in jinja2
    """
    commands_register[f.__name__] = f
    return f
