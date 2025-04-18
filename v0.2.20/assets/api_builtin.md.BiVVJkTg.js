import{_ as i,c as l,j as n,a,G as t,a2 as o,B as p,o as r}from"./chunks/framework.CvXUOK3E.js";const _=JSON.parse('{"title":"Builtin Dialect","description":"","frontmatter":{},"headers":[],"relativePath":"api/builtin.md","filePath":"api/builtin.md","lastUpdated":null}'),c={name:"api/builtin.md"},d={class:"jldocstring custom-block"},u={class:"jldocstring custom-block"};function m(f,e,b,h,y,g){const s=p("Badge");return r(),l("div",null,[e[6]||(e[6]=n("h1",{id:"Builtin-Dialect",tabindex:"-1"},[a("Builtin Dialect "),n("a",{class:"header-anchor",href:"#Builtin-Dialect","aria-label":'Permalink to "Builtin Dialect {#Builtin-Dialect}"'},"​")],-1)),e[7]||(e[7]=n("p",null,[a("Refer to the "),n("a",{href:"https://mlir.llvm.org/docs/Dialects/Builtin/",target:"_blank",rel:"noreferrer"},"official documentation"),a(" for more details.")],-1)),n("details",d,[n("summary",null,[e[0]||(e[0]=n("a",{id:"Reactant.MLIR.Dialects.builtin.module_-Tuple{}",href:"#Reactant.MLIR.Dialects.builtin.module_-Tuple{}"},[n("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.builtin.module_")],-1)),e[1]||(e[1]=a()),t(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[2]||(e[2]=o(`<p><code>module_</code></p><p>A <code>module</code> represents a top-level container operation. It contains a single <a href="./../LangRef#control-flow-and-ssacfg-regions">graph region</a> containing a single block which can contain any operations and does not have a terminator. Operations within this region cannot implicitly capture values defined outside the module, i.e. Modules are <a href="./../Traits#isolatedfromabove">IsolatedFromAbove</a>. Modules have an optional <a href="./../SymbolsAndSymbolTables">symbol name</a> which can be used to refer to them in operations.</p><p><strong>Example</strong></p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>module {</span></span>
<span class="line"><span>  func.func @foo()</span></span>
<span class="line"><span>}</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/cca721dad2218cee45112e61a971f77a44e57e15/src/mlir/Dialects/Builtin.jl#L16-L34" target="_blank" rel="noreferrer">source</a></p>`,5))]),n("details",u,[n("summary",null,[e[3]||(e[3]=n("a",{id:"Reactant.MLIR.Dialects.builtin.unrealized_conversion_cast-Tuple{Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.builtin.unrealized_conversion_cast-Tuple{Vector{Reactant.MLIR.IR.Value}}"},[n("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.builtin.unrealized_conversion_cast")],-1)),e[4]||(e[4]=a()),t(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[5]||(e[5]=o(`<p><code>unrealized_conversion_cast</code></p><p>An <code>unrealized_conversion_cast</code> operation represents an unrealized conversion from one set of types to another, that is used to enable the inter-mixing of different type systems. This operation should not be attributed any special representational or execution semantics, and is generally only intended to be used to satisfy the temporary intermixing of type systems during the conversion of one type system to another.</p><p>This operation may produce results of arity 1-N, and accept as input operands of arity 0-N.</p><p><strong>Example</strong></p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// An unrealized 0-1 conversion. These types of conversions are useful in</span></span>
<span class="line"><span>// cases where a type is removed from the type system, but not all uses have</span></span>
<span class="line"><span>// been converted. For example, imagine we have a tuple type that is</span></span>
<span class="line"><span>// expanded to its element types. If only some uses of an empty tuple type</span></span>
<span class="line"><span>// instance are converted we still need an instance of the tuple type, but</span></span>
<span class="line"><span>// have no inputs to the unrealized conversion.</span></span>
<span class="line"><span>%result = unrealized_conversion_cast to !bar.tuple_type&lt;&gt;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// An unrealized 1-1 conversion.</span></span>
<span class="line"><span>%result1 = unrealized_conversion_cast %operand : !foo.type to !bar.lowered_type</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// An unrealized 1-N conversion.</span></span>
<span class="line"><span>%results2:2 = unrealized_conversion_cast %tuple_operand : !foo.tuple_type&lt;!foo.type, !foo.type&gt; to !foo.type, !foo.type</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// An unrealized N-1 conversion.</span></span>
<span class="line"><span>%result3 = unrealized_conversion_cast %operand, %operand : !foo.type, !foo.type to !bar.tuple_type&lt;!foo.type, !foo.type&gt;</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/cca721dad2218cee45112e61a971f77a44e57e15/src/mlir/Dialects/Builtin.jl#L59-L92" target="_blank" rel="noreferrer">source</a></p>`,6))])])}const R=i(c,[["render",m]]);export{_ as __pageData,R as default};
