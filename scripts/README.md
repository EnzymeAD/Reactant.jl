# TODO Management Scripts

This directory contains scripts to help manage and replace TODO/FIXME/XXX comments in the Reactant.jl codebase with GitHub issue references.

## Overview

The Reactant.jl codebase contains numerous TODO/FIXME/XXX comments scattered throughout. This set of scripts helps to:
1. Collect and catalog all such comments
2. Categorize them thematically
3. Propose a manageable set of GitHub issues
4. Update the comments to reference the created issues

## Scripts

### 1. `collect_todos.jl`

Scans the entire repository for TODO/FIXME/XXX/HACK comments and generates comprehensive reports.

**Usage:**
```bash
julia scripts/collect_todos.jl
```

**Outputs:**
- `scripts/todos.csv` - CSV file with all TODOs (id, file, line, type, text)
- `scripts/todos_report.md` - Markdown report organized by category

### 2. `categorize_todos.jl`

Analyzes collected TODOs and groups them into thematic categories to create a focused set of proposed GitHub issues.

**Usage:**
```bash
julia scripts/categorize_todos.jl
```

**Prerequisites:** Run `collect_todos.jl` first.

**Outputs:**
- `scripts/proposed_issues.md` - Proposed GitHub issues with grouped TODOs
- `scripts/todo_mapping_template.csv` - Template for mapping TODOs to issue numbers

### 3. `update_todos.jl`

Updates TODO comments in the codebase to reference GitHub issues based on a mapping file.

**Usage:**
```bash
julia scripts/update_todos.jl <mapping_file.csv>
```

**Prerequisites:** 
1. Run `collect_todos.jl` to generate the TODOs catalog
2. Create GitHub issues based on `proposed_issues.md`
3. Create a mapping CSV file with columns: `todo_id,issue_number`

**Example mapping file:**
```csv
todo_id,issue_number
1,2225
2,2225
3,2226
...
```

## Workflow

### Step 1: Collect and Analyze TODOs

```bash
# Collect all TODOs
julia scripts/collect_todos.jl

# Categorize and propose issues
julia scripts/categorize_todos.jl
```

### Step 2: Create GitHub Issues

Review `scripts/proposed_issues.md` and create the proposed GitHub issues manually or via GitHub API. The document suggests 12 thematic issues to replace 216+ individual TODOs:

1. General Improvements (~87 items)
2. Error Handling & Validation (~30 items)
3. Sharding & Multi-Device Support (~24 items)
4. MLIR/XLA Integration (~21 items)
5. Build System (~21 items)
6. Type System (~8 items)
7. Array Operations (~8 items)
8. Test Coverage (~5 items)
9. Performance Optimization (~5 items)
10. GPU/TPU Support (~4 items)
11. Documentation (~2 items)
12. Extension Support (~1 item)

### Step 3: Create Mapping File

After creating the GitHub issues, create a CSV file mapping TODO IDs to issue numbers. You can use `todo_mapping_template.csv` as a starting point:

```bash
# Copy the template
cp scripts/todo_mapping_template.csv scripts/todo_mapping.csv

# Edit todo_mapping.csv and fill in the issue_number column
```

### Step 4: Update the Codebase

```bash
# Update all TODOs to reference the created issues
julia scripts/update_todos.jl scripts/todo_mapping.csv
```

This will modify the source files to add issue references like:
```julia
# TODO: implement this feature (see #2225)
```

## Notes

- Generated report files (`*.csv`, `*_report.md`, `proposed_issues.md`) are excluded from version control via `.gitignore`
- The scripts themselves are version controlled
- Always run `collect_todos.jl` first after pulling new changes to get an updated TODO list
- Review changes carefully before committing updated files

## Contributing

If you add new TODOs to the codebase, consider:
1. Referencing an existing issue if applicable
2. Running these scripts periodically to keep TODO tracking up-to-date
3. Creating specific issues for high-priority TODOs

## Related Issues

- #2224 - Meta-issue for replacing TODOs with GitHub issues
