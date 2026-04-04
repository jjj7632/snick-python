# snick-python

Python side of the code for the Snickerdoodle portion of the Sparrow Systems project.

This submodule is intended for work that runs on the Zynq ARM processor under Linux and not FPGA logic. Its main purpose is to support SoC side image handling tasks such as image caching, dummy frame testing, and Python processing hooks for the MATLAB to Snickerdoodle pipeline.

## Current goal
Establish SoC image caching on the Snickerdoodle by storing and returning a dummy image/frame independent of request details. This provides a simple test path before full frame buffering and replay support are implemented
