#!/usr/bin/env julia
"""
Script to collect all TODO/FIXME/XXX comments in the Reactant.jl codebase.
Outputs a JSON file that can be used to create GitHub issues and then update the code.
"""

using Dates

struct TodoItem
    file::String
    line::Int
    type::String  # TODO, FIXME, XXX, HACK
    text::String
    context::String  # a few lines of surrounding context
end

function extract_todos(root_dir::String)
    todos = TodoItem[]
    
    # Patterns to search for
    patterns = [r"\bTODO\b", r"\bFIXME\b", r"\bXXX\b", r"\bHACK\b"]
    
    # File extensions to search
    extensions = [".jl", ".cpp", ".h", ".md", ".toml", ".yaml", ".yml", ".bzl", ".bazelrc"]
    
    # Directories to skip
    skip_dirs = [".git", ".buildkite", "build", ".github/agents", "scripts"]
    
    for (root, dirs, files) in walkdir(root_dir)
        # Filter out skip directories
        filter!(d -> !any(skip -> occursin(skip, joinpath(root, d)), skip_dirs), dirs)
        
        for file in files
            # Special handling for files without extensions
            include_file = any(ext -> endswith(file, ext), extensions) || 
                          file == "BUILD" || file == ".bazelrc"
            
            if !include_file
                continue
            end
            
            filepath = joinpath(root, file)
            
            # Read file
            try
                lines = readlines(filepath)
                for (i, line) in enumerate(lines)
                    # Check for TODO patterns
                    for pattern in patterns
                        if occursin(pattern, line)
                            # Extract the type of marker
                            m = match(pattern, line)
                            if m !== nothing
                                marker_type = m.match
                                
                                # Get context (2 lines before and after)
                                start_line = max(1, i - 2)
                                end_line = min(length(lines), i + 2)
                                context = join(lines[start_line:end_line], "\n")
                                
                                # Clean up the todo text
                                todo_text = strip(line)
                                
                                # Get relative path
                                rel_path = relpath(filepath, root_dir)
                                
                                push!(todos, TodoItem(
                                    rel_path,
                                    i,
                                    marker_type,
                                    todo_text,
                                    context
                                ))
                                break  # Only match once per line
                            end
                        end
                    end
                end
            catch e
                @warn "Failed to process file: $filepath" exception=e
            end
        end
    end
    
    return todos
end

function categorize_todos(todos::Vector{TodoItem})
    categories = Dict{String, Vector{TodoItem}}()
    
    for todo in todos
        # Simple categorization by file path pattern
        category = if occursin(r"src/", todo.file)
            "Core Implementation"
        elseif occursin(r"ext/", todo.file)
            "Extensions"
        elseif occursin(r"test/", todo.file)
            "Tests"
        elseif occursin(r"docs/", todo.file)
            "Documentation"
        elseif occursin(r"deps/", todo.file) || occursin(r"BUILD", todo.file) || occursin(r"\.bzl$", todo.file)
            "Build System"
        elseif occursin(r"benchmark/", todo.file)
            "Benchmarks"
        else
            "Other"
        end
        
        if !haskey(categories, category)
            categories[category] = TodoItem[]
        end
        push!(categories[category], todo)
    end
    
    return categories
end

function export_todos_csv(todos::Vector{TodoItem}, output_file::String)
    open(output_file, "w") do io
        # Header
        println(io, "id,file,line,type,text,issue_number")
        
        # Data
        for (i, t) in enumerate(todos)
            # Escape CSV fields
            text_escaped = replace(t.text, "\"" => "\"\"")
            println(io, "$i,\"$(t.file)\",$(t.line),$(t.type),\"$text_escaped\",")
        end
    end
    
    println("Exported $(length(todos)) TODOs to $output_file")
end

function export_todos_markdown(categories::Dict{String, Vector{TodoItem}}, output_file::String)
    open(output_file, "w") do io
        println(io, "# TODO/FIXME Items in Reactant.jl")
        println(io)
        println(io, "This document lists all TODO/FIXME/XXX/HACK items found in the codebase.")
        println(io, "Each item should be converted into a GitHub issue.")
        println(io)
        
        total_items = sum(length(items) for items in values(categories))
        println(io, "**Total items**: $total_items")
        println(io)
        
        for (category, items) in sort(collect(categories), by=x->x[1])
            println(io, "## $category ($(length(items)) items)")
            println(io)
            
            for (i, item) in enumerate(items)
                println(io, "### Item $i")
                println(io, "- **File**: `$(item.file):$(item.line)`")
                println(io, "- **Type**: $(item.type)")
                println(io, "- **Text**: `$(item.text)`")
                println(io, "- **Issue**: _[To be created]_")
                println(io)
                println(io, "```")
                println(io, item.context)
                println(io, "```")
                println(io)
            end
        end
        
        println(io, "---")
        println(io, "Generated on $(Dates.now())")
    end
    
    println("Exported categorized TODOs to $output_file")
end

function main()
    root_dir = dirname(dirname(@__FILE__))
    println("Scanning $root_dir for TODO items...")
    
    todos = extract_todos(root_dir)
    println("Found $(length(todos)) TODO items")
    
    # Create scripts directory if it doesn't exist
    scripts_dir = joinpath(root_dir, "scripts")
    mkpath(scripts_dir)
    
    # Export to CSV
    csv_file = joinpath(scripts_dir, "todos.csv")
    export_todos_csv(todos, csv_file)
    
    # Export to Markdown
    categories = categorize_todos(todos)
    md_file = joinpath(scripts_dir, "todos_report.md")
    export_todos_markdown(categories, md_file)
    
    # Print summary
    println("\nSummary by category:")
    for (category, items) in sort(collect(categories), by=x->x[1])
        println("  $category: $(length(items)) items")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
