#!/usr/bin/env python
import sys, os
from isipedia.textgenerator import main
sys.path.insert(0, os.path.curdir) # otherwise custom.py does not load
main()