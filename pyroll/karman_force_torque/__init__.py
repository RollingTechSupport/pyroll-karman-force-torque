from . import roll_pass

VERSION = "3.1.0a1"


import importlib.util

REPORT_INSTALLED = bool(importlib.util.find_spec("pyroll.report"))

if REPORT_INSTALLED:
    from . import report
    import pyroll.report

    pyroll.report.plugin_manager.register(report)
