#!/usr/bin/env julia
"""
Script to intelligently categorize and group TODOs into proposed GitHub issues.

This script analyzes TODOs and groups them thematically to create a manageable
number of focused tracking issues.
"""

using Dates

struct TodoItem
    id::Int
    file::String
    line::Int
    type::String
    text::String
end

function load_todos(csv_file::String)
    todos = TodoItem[]
    
    lines = readlines(csv_file)
    # Note: This uses simple CSV parsing since we generate this CSV ourselves
    # and control the format. The text field may contain commas but we use limit
    # to handle this correctly.
    for line in lines[2:end]  # Skip header
        if isempty(strip(line))
            continue
        end
        
        # Simple CSV parser
        parts = split(line, ','; limit=6)
        if length(parts) < 5
            continue
        end
        
        id = parse(Int, strip(parts[1]))
        file = strip(parts[2], '"')
        line_num = parse(Int, strip(parts[3]))
        todo_type = strip(parts[4])
        text = strip(parts[5], '"')
        
        push!(todos, TodoItem(id, file, line_num, todo_type, text))
    end
    
    return todos
end

function categorize_by_theme(todos::Vector{TodoItem})
    # Define thematic categories based on content analysis
    categories = Dict{String, Vector{TodoItem}}()
    
    for todo in todos
        text_lower = lowercase(todo.text)
        file_lower = lowercase(todo.file)
        
        # Categorize based on content
        category = if occursin(r"shard|distributed|multi-device|ifrt|pjrt", text_lower)
            "Sharding & Multi-Device Support"
        elseif occursin(r"error|bounds|check|exception|assert", text_lower)
            "Error Handling & Validation"
        elseif occursin(r"document|doc", text_lower)
            "Documentation"
        elseif occursin(r"test|gradient", text_lower) && occursin("test", file_lower)
            "Test Coverage"
        elseif occursin(r"implement|support|add ", text_lower) && occursin("ext/", file_lower)
            "Extension Support"
        elseif occursin(r"optimi[zs]e|performance|fast|efficient", text_lower)
            "Performance Optimization"
        elseif occursin(r"mlir|dialect|llvm|xla", text_lower)
            "MLIR/XLA Integration"
        elseif occursin(r"toolchain|build|bazel|gcc|clang", text_lower) || occursin("deps/", file_lower)
            "Build System"
        elseif occursin(r"cuda|gpu|tpu", text_lower)
            "GPU/TPU Support"
        elseif occursin(r"type|infer|speciali[zs]e", text_lower)
            "Type System"
        elseif occursin(r"matrix|linear algebra|factori[zs]ation|cholesky|lu", text_lower)
            "Linear Algebra"
        elseif occursin(r"array|tensor|buffer", text_lower)
            "Array Operations"
        else
            "General Improvements"
        end
        
        if !haskey(categories, category)
            categories[category] = TodoItem[]
        end
        push!(categories[category], todo)
    end
    
    return categories
end

function generate_issue_proposals(categories::Dict{String, Vector{TodoItem}}, output_file::String)
    open(output_file, "w") do io
        println(io, "# Proposed GitHub Issues for TODO Items")
        println(io)
        println(io, "This document proposes a structured set of GitHub issues to replace the")
        println(io, "TODO/FIXME/XXX comments in the codebase. Each proposed issue groups related")
        println(io, "TODOs thematically.")
        println(io)
        println(io, "Generated on: $(Dates.now())")
        println(io)
        println(io, "## Summary")
        println(io)
        println(io, "Total TODOs: $(sum(length(items) for items in values(categories)))")
        println(io, "Proposed issues: $(length(categories))")
        println(io)
        
        # Sort categories by number of items (descending)
        sorted_categories = sort([(k, v) for (k, v) in categories]; by=x->-length(x[2]))
        
        issue_num = 1
        for (category, items) in sorted_categories
            println(io, "---")
            println(io)
            println(io, "## Proposed Issue $issue_num: $category")
            println(io)
            println(io, "**Count**: $(length(items)) TODOs")
            println(io)
            println(io, "### Issue Description")
            println(io)
            println(io, "This issue tracks $(length(items)) TODO items related to $category.")
            println(io, "Below is the list of specific items that need to be addressed:")
            println(io)
            
            # Group by file for better readability
            by_file = Dict{String, Vector{TodoItem}}()
            for item in items
                if !haskey(by_file, item.file)
                    by_file[item.file] = TodoItem[]
                end
                push!(by_file[item.file], item)
            end
            
            println(io, "### TODO Items")
            println(io)
            # Sort files by name
            sorted_files = sort(collect(by_file); by=x->x[1])
            for (file, file_items) in sorted_files
                println(io, "#### `$file`")
                println(io)
                for item in sort(file_items; by=x->x.line)
                    println(io, "- Line $(item.line) (ID: $(item.id)): $(item.text)")
                end
                println(io)
            end
            
            issue_num += 1
        end
        
        println(io, "---")
        println(io)
        println(io, "## Next Steps")
        println(io)
        println(io, "1. Review and approve these proposed issues")
        println(io, "2. Create the GitHub issues manually or via API")
        println(io, "3. Create a mapping file (CSV) with columns: todo_id,issue_number")
        println(io, "4. Run `julia scripts/update_todos.jl mapping.csv` to update the code")
    end
    
    println("Generated issue proposals in $output_file")
end

function generate_mapping_template(categories::Dict{String, Vector{TodoItem}}, output_file::String)
    open(output_file, "w") do io
        println(io, "todo_id,issue_number,category")
        
        # Sort categories by number of items (descending)
        sorted_categories = sort([(k, v) for (k, v) in categories]; by=x->-length(x[2]))
        
        issue_num = 1
        for (category, items) in sorted_categories
            for item in items
                println(io, "$(item.id),,\"$category\"")
            end
            issue_num += 1
        end
    end
    
    println("Generated mapping template in $output_file")
end

function main()
    root_dir = dirname(dirname(@__FILE__))
    scripts_dir = joinpath(root_dir, "scripts")
    
    # Load TODOs
    todos_file = joinpath(scripts_dir, "todos.csv")
    if !isfile(todos_file)
        error("TODOs file not found. Run collect_todos.jl first.")
    end
    
    println("Loading TODOs from $todos_file...")
    todos = load_todos(todos_file)
    println("Loaded $(length(todos)) TODOs")
    
    # Categorize
    println("Categorizing TODOs by theme...")
    categories = categorize_by_theme(todos)
    println("Created $(length(categories)) thematic categories")
    
    # Generate proposals
    proposals_file = joinpath(scripts_dir, "proposed_issues.md")
    generate_issue_proposals(categories, proposals_file)
    
    # Generate mapping template
    mapping_file = joinpath(scripts_dir, "todo_mapping_template.csv")
    generate_mapping_template(categories, mapping_file)
    
    # Print summary
    println("\nCategory Summary:")
    # Sort categories by number of items (descending)
    sorted_categories = sort([(k, v) for (k, v) in categories]; by=x->-length(x[2]))
    for (category, items) in sorted_categories
        println("  $(rpad(category, 40)): $(length(items)) items")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
