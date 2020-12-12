import os, sys, logging

# To use isipedia.web, the virtual environment must be inside isipedia.org
isipedia_org = os.path.dirname(sys.prefix)
scripts_folder = os.path.join(isipedia_org, 'scripts')
sys.path.insert(0, scripts_folder)

try:
    from process_articles import *
except ImportError:
    logging.warning('Is isipedia.org virtual environment activated ?')
    raise

if __name__ == '__main__':
    main()
