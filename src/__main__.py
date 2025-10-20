#!/usr/bin/env python3
"""
__main__.py for autonomous research agent package

Allows running the package with: python -m src --args
"""

import asyncio
from .main import main

if __name__ == "__main__":
    asyncio.run(main())