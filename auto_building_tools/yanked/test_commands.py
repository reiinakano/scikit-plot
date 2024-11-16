"""
# test_commands.py
"""
# import sys
# import pytest
# from setuptools.command.test import test as TestCommand

# class PyTest(TestCommand):
#     def finalize_options(self):
#         TestCommand.finalize_options(self)
#         self.test_args = []
#         self.test_suite = True

#     def run_tests(self):
#         errcode = pytest.main(self.test_args)
#         sys.exit(errcode)