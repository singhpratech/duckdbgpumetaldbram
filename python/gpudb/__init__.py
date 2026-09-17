"""gpudb — transparent GPU execution for DuckDB through a thin client wrapper.

    import gpudb
    con = gpudb.connect()                 # same surface as duckdb.connect
    con.execute("SELECT k, sum(v) FROM t GROUP BY k").fetchall()

A SELECT whose shape the resident GROUP BY operators cover, over a table that
is resident on the device, is rewritten before DuckDB plans it (DuckDB's own
parser through json_serialize_sql; docs/TRANSPARENT_DESIGN.md §3). Every other
statement runs unchanged. Residency is automatic (§5): the first sighting of
a shape schedules an upload that runs only while the connection is idle.
"""
from .connection import Connection, connect, GPUDB_EXTENSION_ENV

__all__ = ["Connection", "connect", "GPUDB_EXTENSION_ENV"]
__version__ = "0.7.0.dev0"
