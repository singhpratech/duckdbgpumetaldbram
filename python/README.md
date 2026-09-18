# gpudb — plain DuckDB SQL on the GPU

```python
import gpudb
con = gpudb.connect("my.duckdb")          # same surface as duckdb.connect
con.execute("SELECT k, sum(v) FROM t GROUP BY k").fetchall()
```

Ordinary DuckDB SQL, unchanged: statements the GPU answers faster are answered
on the device, everything else runs on DuckDB exactly as before, and the rows,
names and types are identical either way. `con.last_rewrite()` says what
happened to the last statement.

The wrapper needs the `gpudb` DuckDB extension. It looks, in order, for an
explicit `extension=` path, the `GPUDB_EXTENSION_PATH` environment variable, a
local build next to a source checkout, and finally the extension installed in
DuckDB itself (`INSTALL gpudb FROM community; LOAD gpudb;`).

Settings: `residency` (`background` | `eager` | `manual`), `floor_rows`,
`memory_budget` (e.g. `"16GB"`), `thresholds`. Design, limits and measurements:
`docs/TRANSPARENT_DESIGN.md`, `KNOWN_ISSUES.md` and `BENCHMARK.md` in the repository.
