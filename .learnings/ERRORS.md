## [ERR-20260418-001] save_dataframe_argument_order

**Logged**: 2026-04-18T00:00:00Z
**Priority**: medium
**Status**: pending
**Area**: backend

### Summary

The first full experiment run failed because a `Path` object was passed as the first argument to `save_dataframe`, which expects a `pandas.DataFrame` first and a path second.

### Error
```
AttributeError: 'WindowsPath' object has no attribute 'to_csv'
```

### Context
- Operation attempted: full experiment execution via `run_experiments.py`
- Failing location: `src/digits_project/experiment.py` inside `_save_selected_outputs`
- Cause: argument order was reversed in one call site while the helper signature was correct

### Suggested Fix
Keep helper signatures narrow and check all helper call sites when the first full integration run fails at I/O boundaries.

### Metadata
- Reproducible: yes
- Related Files: src/digits_project/experiment.py, src/digits_project/reporting.py

---