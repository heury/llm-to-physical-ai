import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.modules['model.model_minimind'] = importlib.import_module('model.model_minimind_linear')
target = os.path.abspath(sys.argv.pop(1))
os.chdir(os.path.dirname(target))
__file__ = target
exec(open(target).read())
