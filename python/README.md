# gpudb — plain DuckDB SQL on the GPU

```python
import gpudb
con = gpudb.connect("my.duckdb")          # same surface as duckdb.connect
con.execute("SELECT k, sum(v) FROM t GROUP BY k").fetchall()
```

Ordinary DuckDB SQL, unchanged: statements the GPU answers faster are answered
on the device, everything else runs on DuckDB exactly as before, and the rows,
names and types are identical either way. `con.last_rewrite()` says what
happened to the last statement, including a `detail` line for one that ran on
DuckDB because a threshold said so.

Some shapes are close enough to the line that only measuring in your process
can say. Those keep running on DuckDB while the wrapper times the GPU form
beside them on a cursor of its own, and move to the GPU only once they have
measurably won twice — your own statements are never the experiment.
`con.probation()` shows what is being tried and what it measured.

The wrapper needs the `gpudb` DuckDB extension. It looks, in order, for an
explicit `extension=` path, the `GPUDB_EXTENSION_PATH` environment variable, a
local build next to a source checkout, and finally the extension installed in
DuckDB itself (`INSTALL gpudb FROM community; LOAD gpudb;`).

Settings: `residency` (`background` | `eager` | `manual`), `floor_rows`,
`memory_budget` (e.g. `"16GB"`), `thresholds`. Design, limits and measurements:
`docs/TRANSPARENT_DESIGN.md`, `KNOWN_ISSUES.md` and `BENCHMARK.md` in the repository.
