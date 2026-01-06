// Copyright 2021 Google LLC
// Copyright 2023 Valentin Churavy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <functional>
#include <iostream>
#include <optional>
#include <regex>

#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Region.h"
#include "mlir/TableGen/SideEffects.h"
#include "mlir/TableGen/Trait.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatCommon.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

namespace {

llvm::cl::opt<bool> ExplainMissing(
    "explain-missing",
    llvm::cl::desc("Print the reason for skipping operations from output"));
llvm::cl::opt<std::string>
    DialectName("dialect-name",
                llvm::cl::desc("Override the inferred dialect name, used as "
                               "the name for the generated Julia module."),
                llvm::cl::value_desc("dialect"));

using namespace mlir;
using namespace mlir::tblgen;

/// Returns true if the SameArgumentAndResultTypes trait can be used to infer
/// result types of the given operation.
static bool hasSameArgumentAndResultTypes(const Operator &op) {
  return op.getTrait("::mlir::OpTrait::SameOperandsAndResultType") &&
         op.getNumVariableLengthResults() == 0;
}

/// Returns true if the FirstAttrDerivedResultType trait can be used to infer
/// result types of the given operation.
static bool hasFirstAttrDerivedResultTypes(const Operator &op) {
  return op.getTrait("::mlir::OpTrait::FirstAttrDerivedResultType") &&
         op.getNumVariableLengthResults() == 0;
}

/// Returns true if the InferTypeOpInterface can be used to infer result types
/// of the given operation.
static bool hasInferTypeInterface(const Operator &op) {
  return op.getTrait("::mlir::InferTypeOpInterface::Trait") &&
         op.getNumRegions() == 0;
}

/// Returns true if there is a trait or interface that can be used to infer
/// result types of the given operation.
static bool canInferType(const Operator &op) {
  return hasSameArgumentAndResultTypes(op) ||
         hasFirstAttrDerivedResultTypes(op) || hasInferTypeInterface(op);
}

std::string formatDescription(mlir::tblgen::Operator op) {
  std::string description;
  description = op.getDescription().str();
  size_t pos = 0;
  while (description[pos] == '\n')
    ++pos;
  size_t leading_spaces = 0;
  while (description[pos++] == ' ')
    ++leading_spaces;
  if (leading_spaces) {
    std::string leading_spaces_str;
    for (size_t i = 0; i < leading_spaces; ++i)
      leading_spaces_str += "[ ]";
    description = std::regex_replace(
        description, std::regex("\n" + leading_spaces_str), "\n");
  }
  description = std::regex_replace(description, std::regex(R"(\\)"), R"(\\)");
  description = std::regex_replace(description, std::regex("(['\"$])"), "\\$1");
  description = std::regex_replace(
      description, std::regex("(^|\n)(Example|Syntax):"), "$1# $2");

  // remove trailing whitespaces and newlines
  while (std::isspace(description.back())) {
    description.pop_back();
  }
  return description;
}

std::string getDialectName(llvm::ArrayRef<const llvm::Record *> op_defs) {
  mlir::tblgen::Operator any_op(op_defs.front());
  assert(
      std::all_of(op_defs.begin(), op_defs.end(), [&any_op](llvm::Record *op) {
        return mlir::tblgen::Operator(op).getDialectName() ==
               any_op.getDialectName();
      }));
  std::string dialect_name;
  if (DialectName.empty()) {
    dialect_name = any_op.getDialectName().str();
  } else {
    dialect_name = DialectName;
  }
  return dialect_name;
}

std::string sanitizeName(std::string name,
                         std::optional<std::string> modulename = std::nullopt) {
  // check if name starts with digit:
  if (std::isdigit(name[0])) {
    name = "_" + name;
  }
  // check if name colides with Julia keywords, generated module name, or
  // "location": https://docs.julialang.org/en/v1/base/base/#Keywords
  std::vector<std::string> reservedKeywords = {
      "include", "location", "baremodule", "begin",  "break",    "catch",
      "const",   "continue", "do",         "else",   "elseif",   "end",
      "export",  "false",    "finally",    "for",    "function", "global",
      "if",      "import",   "let",        "local",  "macro",    "module",
      "public",  "quote",    "return",     "struct", "true",     "try",
      "using",   "while"};
  if (modulename.has_value()) {
    reservedKeywords.push_back(modulename.value());
  }
  if (std::find(reservedKeywords.begin(), reservedKeywords.end(), name) !=
      reservedKeywords.end()) {
    name = name + "_";
  }
  // replace all .'s with _'s
  std::replace(name.begin(), name.end(), '.', '_');
  std::replace(name.begin(), name.end(), '-', '_');
  return name;
}

} // namespace

extern bool disableModuleWrap;

bool emitOpTableDefs(const llvm::RecordKeeper &recordKeeper,
                     llvm::raw_ostream &os) {
  llvm::ArrayRef<const llvm::Record *> opdefs =
      recordKeeper.getAllDerivedDefinitionsIfDefined("Op");

  const char *moduleTemplate;
  if (disableModuleWrap) {
    moduleTemplate =
        R"(import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API

{0}
)";
  } else {
    moduleTemplate = R"(module {0}
using ...IR
import ...IR: NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API

{1}
end # {0}
)";
  }

  const char *functiontemplate = R"(
{3}
function {0}({1}location=Location())
    {2}
end
)";      // 0: functionname, 1: functionarguments, 2: functionbody
  const char *functionbodytemplate = R"(op_ty_results = IR.Type[{0}]
    operands = Value[{1}]
    owned_regions = Region[{2}]
    successors = Block[{3}]
    attributes = NamedAttribute[{4}]
    {5}
    create_operation(
        "{6}", location;
        operands, owned_regions, successors, attributes,
        results={7},
        result_inference={8}
    ))"; // 0: results, 1: operands, 2: owned_regions, 3: successors, 4:
         // attributes, 5: optionals, 6: opname, 7: results expression, 8:
         // result_inference

  std::string modulecontents = "";

  std::string modulename;
  if (!DialectName.empty()) {
    modulename = DialectName;
  } else {
    modulename = getDialectName(opdefs);
  }

  for (const auto *def : opdefs) {
    mlir::tblgen::Operator op(*def);

    std::string operandarguments = "";
    std::string operandcontainer = "";
    std::string optionals = "";

    auto opname = op.getOperationName();
    auto functionname = opname.substr(op.getDialectName().str().length() +
                                      1); // get rid of "dialect." prefix.
    functionname = sanitizeName(functionname, modulename);

    std::string description = "";
    if (op.hasDescription()) {
      description = "\"\"\"\n`" + functionname + "`\n" + formatDescription(op) +
                    "\n\"\"\"";
    }
    bool inferrable = canInferType(op);

    bool alreadykeyword =
        false; // set to true when first optional argument is encountered. This
               // is used to insert a single semicolon (;) instead of a comma
               // (,) as separator between positional and keyword arguments.
    for (int i = 0; i < op.getNumOperands(); i++) {
      const auto &named_operand = op.getOperand(i);
      std::string defaultvalue = "";
      std::string operandname = named_operand.name.str();
      if (operandname.empty()) {
        operandname = "operand_" + std::to_string(i);
      }
      operandname = sanitizeName(operandname);

      std::string type = "Value";

      bool optional = named_operand.isOptional();
      bool variadic = named_operand.isVariadic();

      if (variadic) {
        type = "Vector{" + type + "}";
      }

      std::string separator = ", ";
      if (optional) {
        optionals += llvm::formatv(R"(!isnothing({0}) && push!(operands, {0}{1})
    )",
                                   operandname, (variadic ? "..." : ""));
        type = "Union{Nothing, " + type + "}";
        defaultvalue = "=nothing";

        if (!alreadykeyword) {
          alreadykeyword = true;
          separator = "; ";
        }
      } else {
        operandcontainer += operandname + (variadic ? "..." : "") + ", ";
        separator =
            (!alreadykeyword && i == op.getNumOperands() - 1) ? "; " : ", ";
      }

      operandarguments += operandname + defaultvalue + "::" + type + separator;
    }
    if (operandarguments == "") {
      operandarguments = "; ";
    }

    if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
      std::string operandsegmentsizes = "";
      for (int i = 0; i < op.getNumOperands(); i++) {
        const auto &named_operand = op.getOperand(i);
        std::string operandname = named_operand.name.str();
        if (operandname.empty()) {
          operandname = "operand_" + std::to_string(i);
        }
        if (named_operand.isOptional()) {
          operandsegmentsizes += "Int(!isnothing(" + operandname + ")), ";
          continue;
        }
        operandsegmentsizes += named_operand.isVariadic()
                                   ? "length(" + operandname + "), "
                                   : "1, ";
      }
      optionals +=
          llvm::formatv(R"(push!(attributes, operandsegmentsizes([{0}]))
    )",
                        operandsegmentsizes);
    }

    std::string resultarguments = "";
    std::string resultcontainer = "";
    for (int i = 0; i < op.getNumResults(); i++) {
      const auto &named_result = op.getResult(i);
      std::string defaultvalue = "";
      std::string resultname = named_result.name.str();
      if (resultname.empty()) {
        resultname = "result_" + std::to_string(i);
      }
      resultname = sanitizeName(resultname);
      std::string type = "IR.Type";

      bool optional = named_result.isOptional() || inferrable;
      bool variadic = named_result.isVariadic();

      if (variadic) {
        type = "Vector{" + type + "}";
      }

      if (optional) {
        optionals +=
            llvm::formatv(R"(!isnothing({0}) && push!(op_ty_results, {0}{1})
    )",
                          resultname, (variadic ? "..." : ""));
        type = "Union{Nothing, " + type + "}";
        defaultvalue = "=nothing";
      } else {
        resultcontainer += resultname + (variadic ? "..." : "") + ", ";
      }
      resultarguments += resultname + defaultvalue + "::" + type + ", ";
    }

    std::string resultsexpression =
        (inferrable ? "(length(op_ty_results) == 0 ? nothing : op_ty_results)"
                    : "op_ty_results");
    std::string resultinference =
        (inferrable ? "(length(op_ty_results) == 0 ? true : false)" : "false");

    std::string attributearguments = "";
    std::string attributecontainer = "";
    for (int i = 0; i < op.getNumAttributes(); i++) {
      const auto &named_attr = op.getAttribute(i);

      // Derived attributes are never materialized and don't have to be
      // specified.
      if (named_attr.attr.isDerivedAttr())
        continue;

      std::string defaultvalue = "";
      std::string attributename = named_attr.name.str();
      assert(!attributename.empty() &&
             "expected NamedAttribute to have a name");
      std::string sanitizedname = sanitizeName(attributename);

      bool optional =
          named_attr.attr.isOptional() || named_attr.attr.hasDefaultValue();

      if (optional) {
        optionals += llvm::formatv(
            R"(!isnothing({1}) && push!(attributes, namedattribute("{0}", {1}))
    )",
            attributename, sanitizedname);
        defaultvalue = "=nothing";
      } else {
        attributecontainer += "namedattribute(\"" + attributename + "\", " +
                              sanitizedname + "), ";
      }
      attributearguments += sanitizedname + defaultvalue + ", ";
    }

    std::string regionarguments = "";
    std::string regioncontainer = "";
    for (size_t i = 0; i < op.getNumRegions(); i++) {
      const auto &named_region = op.getRegion(i);
      std::string defaultvalue = "";
      std::string regionname = named_region.name.str();
      if (regionname.empty()) {
        regionname = "region_" + std::to_string(i);
      }
      regionname = sanitizeName(regionname);
      std::string type = "Region";

      bool variadic = named_region.isVariadic();

      if (variadic) {
        type = "Vector{" + type + "}";
      }

      regioncontainer += regionname + (variadic ? "..." : "") + ", ";
      regionarguments += regionname + defaultvalue + "::" + type + ", ";
    }

    std::string successorarguments = "";
    std::string successorcontainer = "";
    for (size_t i = 0; i < op.getNumSuccessors(); i++) {
      const auto &named_successor = op.getSuccessor(i);
      std::string defaultvalue = "";
      std::string successorname = named_successor.name.str();
      if (successorname.empty()) {
        successorname = "successor_" + std::to_string(i);
      }
      successorname = sanitizeName(successorname);
      std::string type = "Block";

      bool variadic = named_successor.isVariadic();
      if (variadic) {
        type = "Vector{" + type + "}";
      }

      successorcontainer += successorname + (variadic ? "..." : "") + ", ";
      successorarguments += successorname + defaultvalue + "::" + type + ", ";
    }

    std::string arguments = operandarguments + resultarguments +
                            attributearguments + regionarguments +
                            successorarguments;
    std::string functionbody =
        llvm::formatv(functionbodytemplate, resultcontainer, operandcontainer,
                      regioncontainer, successorcontainer, attributecontainer,
                      optionals, opname, resultsexpression, resultinference);

    modulecontents += llvm::formatv(functiontemplate, functionname, arguments,
                                    functionbody, description);
  }

  if (disableModuleWrap) {
    os << llvm::formatv(moduleTemplate, modulecontents);
  } else {
    os << llvm::formatv(moduleTemplate, modulename, modulecontents);
  }

  return false;
}
