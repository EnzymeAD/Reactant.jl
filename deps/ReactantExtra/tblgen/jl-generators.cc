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

#include <cctype>
#include <optional>
#include <regex>
#include <string>

#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Region.h"
#include "mlir/TableGen/Trait.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
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
  size_t leadingSpaces = 0;
  while (description[pos++] == ' ')
    ++leadingSpaces;
  if (leadingSpaces) {
    std::string leadingSpacesStr;
    for (size_t i = 0; i < leadingSpaces; ++i)
      leadingSpacesStr += "[ ]";
    description = std::regex_replace(description,
                                     std::regex("\n" + leadingSpacesStr), "\n");
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

std::string getDialectName(llvm::ArrayRef<const llvm::Record *> opDefs) {
  mlir::tblgen::Operator anyOp(opDefs.front());
  assert(std::all_of(opDefs.begin(), opDefs.end(),
                     [&anyOp](const llvm::Record *op) {
                       return mlir::tblgen::Operator(op).getDialectName() ==
                              anyOp.getDialectName();
                     }));
  std::string dialectName;
  if (DialectName.empty()) {
    dialectName = anyOp.getDialectName().str();
  } else {
    dialectName = DialectName;
  }
  return dialectName;
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

std::string emitEnum(EnumAttr e, std::string dialect) {
  auto base = e.getBaseAttrClass();
  auto enumName = e.getEnumClassName().str();
  auto name = sanitizeName(enumName);
  auto juliaEnum = "@enumx " + name + ' ';
  auto mlirAttributeDef = "IR.Attribute(e::" + name + ") =  ";
  auto isSpecialized = e.genSpecializedAttr();
  if (!isSpecialized) { // parse the attribute using the name
    for (auto c : e.getAllCases()) {
      juliaEnum += sanitizeName(c.getStr().str()) +
                   ' '; // TODO: handle the case where a enum variant name is a
                        // reserved keyword
    }

    mlirAttributeDef += llvm::formatv(
          R"(parse(Attribute,"#{0}.{1}<$(string(e))>"))", dialect, enumName);
  } else {
    for (auto c : e.getAllCases()) {
      juliaEnum += sanitizeName(c.getStr().str()) + '=' +
                   std::to_string(c.getValue()) + ' ';
    }
    mlirAttributeDef = "(Int(e))";
  }

  return juliaEnum + "\n\n" + mlirAttributeDef + "\n\n";
}

const llvm::StringMap<std::string> cppToJuliaTypeMap = {
    {"int64_t", "Int"},
    {"uint64_t", "Int"},
    {"bool", "Bool"},
    {"unsigned", "Int"},
    {"Type", "IR.Type"},
    {"FunctionType", "IR.Type"},
    {"APInt", "Int64"}, // TODO: add support in reactant to AP
    {"Attribute", "IR.Attribute"},
    {"StringRef", "String"},
    {"ArrayAttr", "Vector{Attribute}"},
    {"FlatSymbolRefAttr", "IR.FlatSymbol"},
    {"ArrayRef<int64_t>", "Vector{Int}"},
};

std::string toPascalCase(std::string s) {
  std::string output = "";
  auto nextUp = true;
  for (auto c : s) {
    if (nextUp) {
      output += std::toupper(c);
      nextUp = false;
      continue;
    }
    if (c == '_') {
      nextUp = true;
      continue;
    }
    output += c;
  }
  return output;
}

// TODO: template support
std::string removeNamespace(std::string s) {
  auto pos = s.rfind("::");
  if (pos >= s.length())
    return s;
  return s.substr(pos + 2);
}

std::string assemblyFormatToJulia(std::string s) {
  auto p = 0;
  auto output = std::string();
  for (auto [i, c] : llvm::enumerate(s)) {
    if (c == '`')
      continue;
    if (c == '$')
      p = i;

    if (p != 0 && c == ' ') {
      auto name = s.substr(p + 1, i - p - 1);
      auto new_name = llvm::formatv("$(s.{})", sanitizeName(name));
      output.append(new_name);
      p = 0;
      continue;
    }

    if (p == 0 && c != ' ')
      output.push_back(c);
  }
  return output;
}

llvm::StringMap<std::string> structMap;
std::optional<std::string> emitStruct(llvm::Record def, std::string dialect) {
  auto assembly = def.getValueAsOptionalString("assemblyFormat");
  auto standardStructAssembly =
      !assembly || *assembly == "`<` struct(params) `>`";
  auto mnemonic = def.getValueAsString("mnemonic").str();
  auto StructName = toPascalCase(mnemonic);
  auto params = def.getValueAsDag("parameters");
  auto predicate = def.getValue("predicate")->getValue()->getAsString();
  auto structDef = "struct " + StructName + '\n';
  auto mlirAttributeDef = "IR.Attribute(s::" + StructName +
                          ") = parse(IR.Attribute,\"#" + dialect + "." +
                          mnemonic;
  if (standardStructAssembly)
    mlirAttributeDef.push_back('<');
  for (auto [arg, name_] :
       llvm::zip(params->getArgs(), params->getArgNames())) {
    auto name = name_->getAsUnquotedString();
    auto sanitizedName = sanitizeName(name);
    auto isArray = false;
    std::string cppType;
    std::optional<std::string> juliaType;
    if (auto init = dyn_cast<llvm::DefInit>(arg)) { // not a cpp type
      auto subDef = init->getDef();
      cppType = subDef->getValueAsString("cppType").str();
      auto type = subDef->getType()->getAsString();
      llvm::StringSwitch<std::function<void()>>(type)
          .Case("APFloatParameter", [&]() { juliaType = "Float64"; })
          .Case("StringRefParameter", [&]() { juliaType = "String"; })
          .Case("EnumParameter",
                [&]() { juliaType = removeNamespace(toPascalCase(cppType)); })
          .Case("ArrayRefParameter",
                [&]() {
                  auto posStart = cppType.find('<') + 1;
                  auto posStop = cppType.rfind('>') - 1;
                  assert(posStart < cppType.length() &&
                         posStop < cppType.length());
                  cppType = cppType.substr(posStart, posStop);
                  cppType.pop_back();
                  cppType = removeNamespace(cppType);
                  isArray = true;
                })
          .Default([&]() {
            llvm::errs() << "unknown pattern : " << type << '\n';
          })();

    } else
      cppType = removeNamespace(arg->getAsUnquotedString());

    if (!juliaType) {
      auto juliaTypeEntry = cppToJuliaTypeMap.find(cppType);
      if (juliaTypeEntry == cppToJuliaTypeMap.end()) {
        llvm::errs() << cppType << '\n';
        return std::nullopt;
      }
      juliaType = juliaTypeEntry->getValue();
    }
    structDef +=
        '\t' + sanitizedName + "::" +
        (isArray ? llvm::formatv("Vector{{{}}", *juliaType) : *juliaType) +
        '\n';
    if (standardStructAssembly)
      mlirAttributeDef +=
          llvm::formatv("{0} = $(s.{1}), ", name, sanitizedName);
  }
  structDef += "end";
  if (standardStructAssembly) {
    mlirAttributeDef.resize(mlirAttributeDef.length() - 2); // remove ,
    mlirAttributeDef += ">";
  } else
    mlirAttributeDef +=
        assemblyFormatToJulia(def.getValueAsString("assemblyFormat").str());
  mlirAttributeDef += "\")";

  structMap.insert({predicate, StructName});
  return structDef + "\n\n" + mlirAttributeDef + "\n\n";
}

bool emitOpTableDefs(const llvm::RecordKeeper &recordKeeper,
                     llvm::raw_ostream &os) {

  llvm::StringMap<EnumAttr> attrMap;
  llvm::ArrayRef<const llvm::Record *> opdefs =
      recordKeeper.getAllDerivedDefinitionsIfDefined("Op");

      //recordKeeper.getDef("anonymous_637")

  std::string moduleName;

  if (!DialectName.empty()) {
    moduleName = DialectName;
  } else {
    moduleName = getDialectName(opdefs);
    DialectName = moduleName;
  }

  llvm::ArrayRef<const llvm::Record *> attrdefs =
      recordKeeper.getAllDerivedDefinitionsIfDefined("Attr");

  auto enumAttr = recordKeeper.getClass("StrAttr");
  recordKeeper.getDef("StrAttr");
  auto attribs = std::string();
  for (auto a : attrdefs) {
    mlir::tblgen::Attribute attr(a);
    if (attr.isDerivedAttr())
      continue;
    auto name = attr.getDefName();
    if (attr.isEnumAttr()) { // Some EnumAttribute don't have an EnumAttrInfo
                             // bound : see VHLO dialect
      EnumAttr enumAttr(a);
      attribs += emitEnum(enumAttr, moduleName) + '\n';
      auto s = enumAttr.getCppNamespace() +
               "::" + enumAttr.getSpecializedAttrClassName();
      attrMap.insert({s.str(), enumAttr});
      continue;
    }

    if (attr.isSubClassOf("EnumAttr")){
      attr.getDef().dump();
      continue;
    }
     

    if (attr.getDef().getValue("attrName")) { // detect "struct" attributes
      auto structAttr = emitStruct(attr.getDef(), moduleName);
      if (!structAttr)
        continue;
      attribs += *structAttr;
    }
  }
  const char *moduleTemplate;
  if (disableModuleWrap) {
    moduleTemplate =
        R"(import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API
using EnumX
{0}
)";
  } else {
    moduleTemplate = R"(module {0}
using ...IR
import ...IR: NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API
using EnumX

{1}
end # {0}
)";
  }

  const char *functionTemplate = R"(
{3}
function {0}({1}location=Location())
    {2}
end
)";      // 0: functionname, 1: functionarguments, 2: functionbody
  const char *functionBodyTemplate = R"(op_ty_results = IR.Type[{0}]
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

  std::string moduleContents = "";

  for (const auto *def : opdefs) {
    mlir::tblgen::Operator op(*def);

    std::string operandArguments = "";
    std::string operandContainer = "";
    std::string optionals = "";

    auto opname = op.getOperationName();
    auto functionName = opname.substr(op.getDialectName().str().length() +
                                      1); // get rid of "dialect." prefix.
    functionName = sanitizeName(functionName, moduleName);

    std::string description = "";
    if (op.hasDescription()) {
      description = "\"\"\"\n`" + functionName + "`\n" + formatDescription(op) +
                    "\n\"\"\"";
    }
    bool inferrable = canInferType(op);

    bool alreadykeyword =
        false; // set to true when first optional argument is encountered. This
               // is used to insert a single semicolon (;) instead of a comma
               // (,) as separator between positional and keyword arguments.
    for (int i = 0; i < op.getNumOperands(); i++) {
      const auto &namedOperand = op.getOperand(i);
      std::string defaultvalue = "";
      std::string operandName = namedOperand.name.str();
      if (operandName.empty()) {
        operandName = "operand_" + std::to_string(i);
      }
      operandName = sanitizeName(operandName);

      std::string type = "Value";

      bool optional = namedOperand.isOptional();
      bool variadic = namedOperand.isVariadic();

      if (variadic) {
        type = "Vector{" + type + "}";
      }

      std::string separator = ", ";
      if (optional) {
        optionals += llvm::formatv(R"(!isnothing({0}) && push!(operands, {0}{1})
    )",
                                   operandName, (variadic ? "..." : ""));
        type = "Union{Nothing, " + type + "}";
        defaultvalue = "=nothing";

        if (!alreadykeyword) {
          alreadykeyword = true;
          separator = "; ";
        }
      } else {
        operandContainer += operandName + (variadic ? "..." : "") + ", ";
        separator =
            (!alreadykeyword && i == op.getNumOperands() - 1) ? "; " : ", ";
      }

      operandArguments += operandName + "::" + type + defaultvalue + separator;
    }
    if (operandArguments == "") {
      operandArguments = "; ";
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
          operandsegmentsizes += "(" + operandname + "==nothing) ? 0 : 1";
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

    std::string resultArguments = "";
    std::string resultContainer = "";
    for (int i = 0; i < op.getNumResults(); i++) {
      const auto &namedResult = op.getResult(i);
      std::string defaultvalue = "";
      std::string resultname = namedResult.name.str();
      if (resultname.empty()) {
        resultname =
            op.getNumResults() == 1 ? "result" : "result_" + std::to_string(i);
      }
      resultname = sanitizeName(resultname);
      std::string type = "IR.Type";

      bool optional = namedResult.isOptional() || inferrable;
      bool variadic = namedResult.isVariadic();

      if (variadic) {
        type = "Union{Vector{" + type + "}, Tuple{Vararg{" + type + "}}}";
      }

      if (optional) {
        optionals +=
            llvm::formatv(R"(!isnothing({0}) && push!(op_ty_results, {0}{1})
    )",
                          resultname, (variadic ? "..." : ""));
        type = "Union{Nothing, " + type + "}";
        defaultvalue = "=nothing";
      } else {
        resultContainer += resultname + (variadic ? "..." : "") + ", ";
      }
      resultArguments += resultname + "::" + type + defaultvalue + ", ";
    }

    std::string resultsexpression =
        (inferrable ? "(isempty(op_ty_results) ? nothing : op_ty_results)"
                    : "op_ty_results");
    std::string resultInference =
        (inferrable ? "isempty(op_ty_results)" : "false");

    std::string attributeArguments = "";
    std::string attributeContainer = "";
    for (int i = 0; i < op.getNumAttributes(); i++) {
      const auto &namedAttr = op.getAttribute(i);
      auto attr = namedAttr.attr;
      // Derived attributes are never materialized and don't have to be
      // specified.
      if (attr.isDerivedAttr())
        continue;

      std::string defaultValue = "";
      std::string attributeName = namedAttr.name.str();

      assert(!attributeName.empty() &&
             "expected NamedAttribute to have a name");

      auto optional = attr.isOptional() || attr.hasDefaultValue();

      std::string VarName = sanitizeName(attributeName);
      std::string pushedExpression;
      std::string varType;

      auto entry = attrMap.find(attr.getStorageType());
      if (entry != attrMap.end()) {
        auto e = entry->getValue();
        auto isSpecialized = e.genSpecializedAttr();
        varType = sanitizeName(e.getEnumClassName().str()) + ".T";
        if (isSpecialized) {
          pushedExpression = llvm::formatv(R"(Int({0}))", VarName);
        } else {
          pushedExpression =
              llvm::formatv(R"(parse(Attribute,"#{0}.{1}<$(string({1}))>"))",
                            DialectName, VarName);
        }
      } else {
        attr = optional ? attr.getBaseAttr() : attr;

        auto attr_entry = cppToJuliaTypeMap.find(attr.getAttrDefName());
        if (attr_entry != cppToJuliaTypeMap.end()) {
          varType = attr_entry->getValue();
          pushedExpression = VarName;
        } else {
          auto fullCppType = attr.getDef()
                                 .getValue("returnType")
                                 ->getValue()
                                 ->getAsUnquotedString();
          auto cppType = removeNamespace(fullCppType);
          cppType.erase(std::remove(cppType.begin(), cppType.end(), ' '),
                        cppType.end());
          auto entry = cppToJuliaTypeMap.find(cppType);
          if (entry != cppToJuliaTypeMap.end()) {
            varType = entry->getValue();
            pushedExpression = VarName;
          } else {
            auto entry = structMap.find(
                attr.getDef().getValue("predicate")->getValue()->getAsString());
            if (entry != structMap.end()) {
              varType = entry->getValue();
              pushedExpression = "Attribute(" + VarName + ")";
            } else {
              pushedExpression = VarName;
              varType = "Any";
            }
          }
        }
      }

      if (optional) {
        optionals += llvm::formatv(
            R"(!isnothing({0}) && push!(attributes, namedattribute("{0}", {1}))
    )",
            attributeName, pushedExpression);
        defaultValue = "=nothing";
        varType = "Union{" + varType + ", Nothing}";
      } else {
        attributeContainer += "namedattribute(\"" + attributeName + "\", " +
                              pushedExpression + "), ";
      }
      attributeArguments += VarName + "::" + varType + defaultValue + ", ";
    }

    std::string regionArguments = "";
    std::string regionContainer = "";
    for (size_t i = 0; i < op.getNumRegions(); i++) {
      const auto &namedRegion = op.getRegion(i);
      std::string defaultvalue = "";
      std::string regionName = namedRegion.name.str();
      if (regionName.empty()) {
        regionName = "region_" + std::to_string(i);
      }
      regionName = sanitizeName(regionName);
      std::string type = "Region";

      bool variadic = namedRegion.isVariadic();

      if (variadic) {
        type = "Vector{" + type + "}";
      }

      regionContainer += regionName + (variadic ? "..." : "") + ", ";
      regionArguments += regionName + "::" + type + defaultvalue + ", ";
    }

    std::string successorArguments = "";
    std::string successorContainer = "";
    for (size_t i = 0; i < op.getNumSuccessors(); i++) {
      const auto &namedSuccessor = op.getSuccessor(i);
      std::string defaultValue = "";
      std::string successorName = namedSuccessor.name.str();
      if (successorName.empty()) {
        successorName = "successor_" + std::to_string(i);
      }
      successorName = sanitizeName(successorName);
      std::string type = "Block";

      bool variadic = namedSuccessor.isVariadic();
      if (variadic) {
        type = "Vector{" + type + "}";
      }

      successorContainer += successorName + (variadic ? "..." : "") + ", ";
      successorArguments += successorName + "::" + type + defaultValue + ", ";
    }

    std::string arguments = operandArguments + resultArguments +
                            attributeArguments + regionArguments +
                            successorArguments;
    std::string functionBody =
        llvm::formatv(functionBodyTemplate, resultContainer, operandContainer,
                      regionContainer, successorContainer, attributeContainer,
                      optionals, opname, resultsexpression, resultInference);

    moduleContents += llvm::formatv(functionTemplate, functionName, arguments,
                                    functionBody, description);
  }

  moduleContents = attribs + moduleContents;

  if (disableModuleWrap) {
    os << llvm::formatv(moduleTemplate, moduleContents);
  } else {
    os << llvm::formatv(moduleTemplate, moduleName, moduleContents);
  }

  return false;
}
