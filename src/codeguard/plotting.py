"""Headless plotting for reports saved to disk."""

import os


def report_pyplot():
    # Matplotlib validates MPLBACKEND during import, before use() can override it.
    # Colab's inherited inline backend may be absent in the analysis environment.
    os.environ['MPLBACKEND'] = 'Agg'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt
