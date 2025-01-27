# -*- coding: utf-8 -*-
"""

**PROJECT DA VINCI**

by Lightning Custom Shop

Created on Sun May 12 18:28:34 2024

@author: mtthl

This module contains all the functions and objects of Project da Vinci.

    Project da Vinci is a LCS project to understand a variety of data via the same math and 
patterns used to understand quantum mechanics and fluid turbulence. This will be based on the 
proposed understanding by Mandelbrot.
    
Version     Date        Description

0.0         2024/05/12  The initial version

0.1         2024/05/13  Added the finance module and the transform_lib module. Made more organized.

"""

#==================================================================================================
#
# Import Modules
#
#==================================================================================================

#
# Import Environment Packages
#
import os, sys
import numpy as np

#
# Import da Vinci Libararies
#
script_dir = os.path.dirname(os.path.realpath(__file__))
lib_dir = os.path.join(script_dir, 'lib')
sys.path.append(lib_dir)

from finance import *
from transform_lib import *



###############################################################################
#
# Objects
#
###############################################################################

