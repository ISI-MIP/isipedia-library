import os, sys, logging

# find out the appropriate paths
found = False
for test in ['isipedia.org', '../isipedia.org', '../../isipedia.org', '../../../isipedia.org']:
    if os.path.exists(test):
        webscript = test+'/scripts'
        sys.path.append(webscript)
        print(webscript, 'appended to python path')
        found = True
        isipedia_org = test
        break
if not found: 
    isipedia_org = None
    logging.warning('isipedia.org was not found')

try:
    from process_articles import *
except ImportError:
    logging.warning('failed import process_articles')
    raise


if __name__ == '__main__':
    main()
