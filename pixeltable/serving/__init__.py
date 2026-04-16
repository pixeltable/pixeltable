"""
Adapters for web serving frameworks.
"""

try:
    from ._fastapi import FastAPIRouter
except ImportError:
    # fastapi is an optional dependency; leave FastAPIRouter undefined if it isn't installed
    pass

__all__ = ['FastAPIRouter']
