# TODO Tracking: Proposed Implementation Plan

**Related Issue:** #2224  
**Generated:** 2026-01-26

## Executive Summary

The Reactant.jl codebase contains **216 TODO/FIXME/XXX comments** across 72 files. These have been analyzed and categorized into 12 thematic groups. This document proposes a practical implementation plan to replace these comments with GitHub issue references.

## Current State

### TODO Distribution
- **Core Implementation** (src/): 124 items
- **Extensions** (ext/): 35 items
- **Build System** (deps/): 23 items
- **Tests** (test/): 23 items
- **Documentation** (docs/): 7 items
- **Benchmarks** (benchmark/): 4 items

### Categorization by Theme
1. **General Improvements**: 87 items
2. **Error Handling & Validation**: 30 items
3. **Sharding & Multi-Device Support**: 24 items
4. **MLIR/XLA Integration**: 21 items
5. **Build System**: 21 items
6. **Type System**: 8 items
7. **Array Operations**: 8 items
8. **Test Coverage**: 5 items
9. **Performance Optimization**: 5 items
10. **GPU/TPU Support**: 4 items
11. **Documentation**: 2 items
12. **Extension Support**: 1 item

## Proposed Approach

### Option A: Create 12 Thematic Tracking Issues (Recommended)

Instead of creating 216 individual issues, create 12 focused tracking issues that group related TODOs. This approach:
- ✅ Reduces issue clutter
- ✅ Provides better organization
- ✅ Makes it easier to track progress on related items
- ✅ Allows for more detailed issue descriptions with context

**Implementation:**
1. Use `scripts/proposed_issues.md` as a template to create 12 GitHub issues
2. Each issue title: "[TODO Tracking] <Category Name>"
3. Issue body includes the list of specific TODO items with file:line references
4. Create a mapping CSV with TODO IDs and issue numbers
5. Run `julia scripts/update_todos.jl mapping.csv` to update all code

**Time estimate:** 30-60 minutes to create issues + 5 minutes to run update script

### Option B: Create Individual Issues for High-Priority TODOs

Create issues only for:
- TODOs marked with specific people (`TODO(@username)`)
- TODOs blocking features (error messages with "Not yet Implemented")
- TODOs with external issue references

**Implementation:**
1. Manually review `scripts/todos_report.md`
2. Create issues for ~20-30 high-priority items
3. Update those specific TODOs
4. Keep remaining TODOs as-is or reference meta-issue #2224

**Time estimate:** 1-2 hours

### Option C: Temporary Solution with Meta-Issue Reference

Update all TODOs to reference the meta-issue #2224 temporarily, then create more specific issues later.

**Pros:**
- ✅ Quick implementation (5 minutes)
- ✅ All TODOs now have an issue reference
- ✅ Can be refined later

**Cons:**
- ❌ Less useful for tracking (all point to same issue)
- ❌ Requires second pass to add specific issues

## Recommended Next Steps

1. **Review & Decide**: Choose Option A (recommended), B, or C above
2. **Create Issues**: 
   - If Option A: Use `scripts/proposed_issues.md` to create 12 issues
   - If Option B: Manually create priority issues
   - If Option C: Skip this step
3. **Create Mapping**: Fill in `scripts/todo_mapping_template.csv` with issue numbers
4. **Update Code**: Run `julia scripts/update_todos.jl <mapping_file>`
5. **Verify**: Check a few files to ensure references were added correctly
6. **Commit**: Commit all updated files

## Tools Available

All scripts are in the `scripts/` directory with documentation in `scripts/README.md`:

- **`collect_todos.jl`**: Scans codebase for TODOs → generates CSV and markdown reports
- **`categorize_todos.jl`**: Groups TODOs by theme → generates proposed issues
- **`update_todos.jl`**: Updates code with issue references based on mapping file

## Example: What the Updated Code Will Look Like

**Before:**
```julia
# TODO: implement sharding
```

**After:**
```julia
# TODO: implement sharding (see #2230)
```

## Considerations

### Some TODOs Already Reference Issues
A few TODOs already reference external issues (e.g., Enzyme-JAX issues). These should be preserved.

### Some TODOs Are in Generated Files
Files in `src/mlir/` may be partially generated. Updates should be reviewed carefully.

### Build System TODOs
TODOs in Bazel/BUILD files and C++ code (deps/ReactantExtra/) should be tracked but may need different formatting.

## Existing Related Issues

Some existing issues may already cover TODO items:
- #693 - Sharding & Multi-Device Execution Meta-Issue
- #2058 - Move `sharding` to type domain  
These could be used for sharding-related TODOs.

## Questions?

Contact: Repository maintainers or comment on issue #2224

---

**Files Generated:**
- `scripts/todos.csv` - Complete list of all TODOs
- `scripts/todos_report.md` - Organized by file/category
- `scripts/proposed_issues.md` - Ready-to-use issue templates
- `scripts/todo_mapping_template.csv` - Template for creating mappings
