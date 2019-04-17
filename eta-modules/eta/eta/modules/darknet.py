#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import logging
import sys

import eta.core.module as etam

from dmc_darknet import run_darknet
# By convention, all ETA modules should use a logger whose name is `__name__`
# to log all messages
logger = logging.getLogger(__name__)


#
# The following class defines the content of the module configuration file.
#
# This class inherits from `BaseModuleConfig`, which handles the parsing of the
# base module settings automatically.
#
# The docstring of this class must contain an `Attributes` section that
# specifies the classes that describe the `data` and `parameters` fields. This
# information is used by the metadata generation tool.
#9



CFG = "/home/nick/Workspace/src/github.com/pjreddie/darknet/cfg/yolov2.cfg"
DATA = "/home/nick/Workspace/src/github.com/pjreddie/darknet/cfg/coco.data"
WEIGHTS = "/home/nick/Workspace/src/github.com/pjreddie/darknet/yolov2.weights"



class DarknetConfig(etam.BaseModuleConfig):

    def __init__(self, d):
        # Call the `BaseModuleConfig` constructor, which parses the optional
        # `base` field
        super(DarknetConfig, self).__init__(d)

        # Parse the `data` field, which is defined by an array of `DataConfig`
        # instances
        self.data = self.parse_object_array(d, "data", DataConfig)

        # Parse the `parameters` field, which is defined by a
        # `ParametersConfig` instance
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(etam.Config):
    '''Data configuration settings.

    Inputs:
        {{input1}} ({{type1}}): {{description1}}
        {{input2}} ({{type2}}): [None] {{description2}}

    Outputs:
        {{output1}} ({{type3}}): {{description3}}
        {{output2}} ({{type4}}): [None] {{description4}}
    '''

    def __init__(self, d):
        # Template for parsing a required input
        self.input_path = self.parse_string(d, "input_path")

        # Template for parsing a required output
        self.output_path = self.parse_string(d, "output_path")


#
# The following class defines the parameters for the module.
#
# The docstring of this class must contain a `Parameters` section that
# describes the parameters supported by the module. This information is used by
# the metadata generation tool.
#
class ParametersConfig(etam.Config):
    '''Parameter configuration settings.

    Parameters:
        {{parameter1}} ({{type5}}): {{description5}}
        {{parameter2}} ({{type6}}): [{{default1}}] {{description6}}
    '''

    def __init__(self, d):
        # Required Parameters
        self.cfg_path = self.parse_string(d, "cfg_path")
        self.weights_path = self.parse_string(d, "weights_path")
        self.data_path = self.parse_string(d, "data_path")


#
# By convention, all modules in the ETA library define a `run()` method that
# parses the command-line arguments, performs base module setup, and then calls
# another method that implements the actual module-specific actions
#
def run(config_path, pipeline_config_path=None):
    '''Run the {{module_name}} module.

    Args:
        config_path: path to a {{ModuleName}}Config file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    # Load the module config
    config = DarknetConfig.from_json(config_path)

    # Perform base module setup via the `eta.core.module.setup()` method
    # provided by the ETA library
    etam.setup(config, pipeline_config_path=pipeline_config_path)

    # Now pass the `config` instance to another private method to perform the
    # actual module computations
    # ..."
    logger.debug("Trying to invoke the darknet code now...")
    run_darknet(config)



if __name__ == "__main__":
    # Pass the command-line arguments to the `run()` method for parsing and
    # processing
    run(*sys.argv[1:])
