#!c:\users\chewlwb\documents\github\cs6400-2021-01-team048\phase_3\venv\scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'ipdb==0.13.7','console_scripts','ipdb3'
__requires__ = 'ipdb==0.13.7'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('ipdb==0.13.7', 'console_scripts', 'ipdb3')()
    )
