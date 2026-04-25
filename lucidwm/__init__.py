"""LucidWM (Layer 2): Single-file algorithm implementations.

Each file is self-contained: imports from Layer 1 (components) and Layer 3 (utils),
but contains all algorithm-specific logic. Files never import from each other.

Run any algorithm directly:
    python -m lucidwm.dreamer_v3_dmc --env-id walker-walk --seed 1 --track
"""

__version__ = "0.1.0"
