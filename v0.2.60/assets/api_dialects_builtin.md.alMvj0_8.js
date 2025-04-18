import{_ as l,C as p,c as r,o as c,j as n,a,a2 as o,G as t,w as i}from"./chunks/framework.B7LmI5qL.js";const R=JSON.parse('{"title":"Builtin Dialect","description":"","frontmatter":{},"headers":[],"relativePath":"api/dialects/builtin.md","filePath":"api/dialects/builtin.md","lastUpdated":null}'),d={name:"api/dialects/builtin.md"},u={class:"jldocstring custom-block"},m={class:"jldocstring custom-block"};function f(y,e,b,_,h,g){const s=p("Badge");return c(),r("div",null,[e[8]||(e[8]=n("h1",{id:"Builtin-Dialect",tabindex:"-1"},[a("Builtin Dialect "),n("a",{class:"header-anchor",href:"#Builtin-Dialect","aria-label":'Permalink to "Builtin Dialect {#Builtin-Dialect}"'},"​")],-1)),e[9]||(e[9]=n("p",null,[a("Refer to the "),n("a",{href:"https://mlir.llvm.org/docs/Dialects/Builtin/",target:"_blank",rel:"noreferrer"},"official documentation"),a(" for more details.")],-1)),n("details",u,[n("summary",null,[e[0]||(e[0]=n("a",{id:"Reactant.MLIR.Dialects.builtin.module_-Tuple{}",href:"#Reactant.MLIR.Dialects.builtin.module_-Tuple{}"},[n("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.builtin.module_")],-1)),e[1]||(e[1]=a()),t(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[3]||(e[3]=o(`<p><code>module_</code></p><p>A <code>module</code> represents a top-level container operation. It contains a single <a href="./../LangRef#control-flow-and-ssacfg-regions">graph region</a> containing a single block which can contain any operations and does not have a terminator. Operations within this region cannot implicitly capture values defined outside the module, i.e. Modules are <a href="./../Traits#isolatedfromabove">IsolatedFromAbove</a>. Modules have an optional <a href="./../SymbolsAndSymbolTables">symbol name</a> which can be used to refer to them in operations.</p><p><strong>Example</strong></p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>module {</span></span>
<span class="line"><span>  func.func @foo()</span></span>
<span class="line"><span>}</span></span></code></pre></div>`,4)),t(s,{type:"info",class:"source-link",text:"source"},{default:i(()=>e[2]||(e[2]=[n("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a334a738456754d42d7c08a3d1994d41f83fcd17/src/mlir/Dialects/Builtin.jl#L16-L34",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),n("details",m,[n("summary",null,[e[4]||(e[4]=n("a",{id:"Reactant.MLIR.Dialects.builtin.unrealized_conversion_cast-Tuple{Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.builtin.unrealized_conversion_cast-Tuple{Vector{Reactant.MLIR.IR.Value}}"},[n("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.builtin.unrealized_conversion_cast")],-1)),e[5]||(e[5]=a()),t(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[7]||(e[7]=o(`<p><code>unrealized_conversion_cast</code></p><p>An <code>unrealized_conversion_cast</code> operation represents an unrealized conversion from one set of types to another, that is used to enable the inter-mixing of different type systems. This operation should not be attributed any special representational or execution semantics, and is generally only intended to be used to satisfy the temporary intermixing of type systems during the conversion of one type system to another.</p><p>This operation may produce results of arity 1-N, and accept as input operands of arity 0-N.</p><p><strong>Example</strong></p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// An unrealized 0-1 conversion. These types of conversions are useful in</span></span>
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
<span class="line"><span>%result3 = unrealized_conversion_cast %operand, %operand : !foo.type, !foo.type to !bar.tuple_type&lt;!foo.type, !foo.type&gt;</span></span></code></pre></div>`,5)),t(s,{type:"info",class:"source-link",text:"source"},{default:i(()=>e[6]||(e[6]=[n("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a334a738456754d42d7c08a3d1994d41f83fcd17/src/mlir/Dialects/Builtin.jl#L59-L92",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})])])}const T=l(d,[["render",f]]);export{R as __pageData,T as default};
