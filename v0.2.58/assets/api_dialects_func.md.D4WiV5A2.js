import{_ as c,C as o,c as p,o as r,j as t,a as s,a2 as l,G as a,w as i}from"./chunks/framework.CHmd-7HT.js";const I=JSON.parse('{"title":"Func Dialect","description":"","frontmatter":{},"headers":[],"relativePath":"api/dialects/func.md","filePath":"api/dialects/func.md","lastUpdated":null}'),u={name:"api/dialects/func.md"},f={class:"jldocstring custom-block"},d={class:"jldocstring custom-block"},m={class:"jldocstring custom-block"},g={class:"jldocstring custom-block"},b={class:"jldocstring custom-block"};function h(_,e,R,y,T,x){const n=o("Badge");return r(),p("div",null,[e[20]||(e[20]=t("h1",{id:"Func-Dialect",tabindex:"-1"},[s("Func Dialect "),t("a",{class:"header-anchor",href:"#Func-Dialect","aria-label":'Permalink to "Func Dialect {#Func-Dialect}"'},"​")],-1)),e[21]||(e[21]=t("p",null,[s("Refer to the "),t("a",{href:"https://mlir.llvm.org/docs/Dialects/Func/",target:"_blank",rel:"noreferrer"},"official documentation"),s(" for more details.")],-1)),t("details",f,[t("summary",null,[e[0]||(e[0]=t("a",{id:"Reactant.MLIR.Dialects.func.call-Tuple{Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.func.call-Tuple{Vector{Reactant.MLIR.IR.Value}}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.func.call")],-1)),e[1]||(e[1]=s()),a(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[3]||(e[3]=l('<p><code>call</code></p><p>The <code>func.call</code> operation represents a direct call to a function that is within the same symbol scope as the call. The operands and result types of the call must match the specified function type. The callee is encoded as a symbol reference attribute named &quot;callee&quot;.</p><p><strong>Example</strong></p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>%2 = func.call @my_add(%0, %1) : (f32, f32) -&gt; f32</span></span></code></pre></div>',4)),a(n,{type:"info",class:"source-link",text:"source"},{default:i(()=>e[2]||(e[2]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/13dc7a41f819f26358c64ee5e6f17d7be57713f5/src/mlir/Dialects/Func.jl#L61-L74",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",d,[t("summary",null,[e[4]||(e[4]=t("a",{id:"Reactant.MLIR.Dialects.func.call_indirect-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.func.call_indirect-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.func.call_indirect")],-1)),e[5]||(e[5]=s()),a(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[7]||(e[7]=l(`<p><code>call_indirect</code></p><p>The <code>func.call_indirect</code> operation represents an indirect call to a value of function type. The operands and result types of the call must match the specified function type.</p><p>Function values can be created with the <a href="/Reactant.jl/v0.2.58/api/dialects/func#funcconstant-constantop"><code>func.constant</code> operation</a>.</p><p><strong>Example</strong></p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>%func = func.constant @my_func : (tensor&lt;16xf32&gt;, tensor&lt;16xf32&gt;) -&gt; tensor&lt;16xf32&gt;</span></span>
<span class="line"><span>%result = func.call_indirect %func(%0, %1) : (tensor&lt;16xf32&gt;, tensor&lt;16xf32&gt;) -&gt; tensor&lt;16xf32&gt;</span></span></code></pre></div>`,5)),a(n,{type:"info",class:"source-link",text:"source"},{default:i(()=>e[6]||(e[6]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/13dc7a41f819f26358c64ee5e6f17d7be57713f5/src/mlir/Dialects/Func.jl#L16-L32",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",m,[t("summary",null,[e[8]||(e[8]=t("a",{id:"Reactant.MLIR.Dialects.func.constant-Tuple{}",href:"#Reactant.MLIR.Dialects.func.constant-Tuple{}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.func.constant")],-1)),e[9]||(e[9]=s()),a(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[11]||(e[11]=l(`<p><code>constant</code></p><p>The <code>func.constant</code> operation produces an SSA value from a symbol reference to a <code>func.func</code> operation</p><p><strong>Example</strong></p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// Reference to function @myfn.</span></span>
<span class="line"><span>%2 = func.constant @myfn : (tensor&lt;16xf32&gt;, f32) -&gt; tensor&lt;16xf32&gt;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// Equivalent generic forms</span></span>
<span class="line"><span>%2 = &quot;func.constant&quot;() { value = @myfn } : () -&gt; ((tensor&lt;16xf32&gt;, f32) -&gt; tensor&lt;16xf32&gt;)</span></span></code></pre></div><p>MLIR does not allow direct references to functions in SSA operands because the compiler is multithreaded, and disallowing SSA values to directly reference a function simplifies this (<a href="./../Rationale/Rationale#multithreading-the-compiler">rationale</a>).</p>`,5)),a(n,{type:"info",class:"source-link",text:"source"},{default:i(()=>e[10]||(e[10]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/13dc7a41f819f26358c64ee5e6f17d7be57713f5/src/mlir/Dialects/Func.jl#L105-L125",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",g,[t("summary",null,[e[12]||(e[12]=t("a",{id:"Reactant.MLIR.Dialects.func.func_-Tuple{}",href:"#Reactant.MLIR.Dialects.func.func_-Tuple{}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.func.func_")],-1)),e[13]||(e[13]=s()),a(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[15]||(e[15]=l(`<p><code>func_</code></p><p>Operations within the function cannot implicitly capture values defined outside of the function, i.e. Functions are <code>IsolatedFromAbove</code>. All external references must use function arguments or attributes that establish a symbolic connection (e.g. symbols referenced by name via a string attribute like SymbolRefAttr). An external function declaration (used when referring to a function declared in some other module) has no body. While the MLIR textual form provides a nice inline syntax for function arguments, they are internally represented as “block arguments” to the first block in the region.</p><p>Only dialect attribute names may be specified in the attribute dictionaries for function arguments, results, or the function itself.</p><p><strong>Example</strong></p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// External function definitions.</span></span>
<span class="line"><span>func.func private @abort()</span></span>
<span class="line"><span>func.func private @scribble(i32, i64, memref&lt;? x 128 x f32, #layout_map0&gt;) -&gt; f64</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// A function that returns its argument twice:</span></span>
<span class="line"><span>func.func @count(%x: i64) -&gt; (i64, i64)</span></span>
<span class="line"><span>  attributes {fruit = &quot;banana&quot;} {</span></span>
<span class="line"><span>  return %x, %x: i64, i64</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// A function with an argument attribute</span></span>
<span class="line"><span>func.func private @example_fn_arg(%x: i32 {swift.self = unit})</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// A function with a result attribute</span></span>
<span class="line"><span>func.func private @example_fn_result() -&gt; (f64 {dialectName.attrName = 0 : i64})</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// A function with an attribute</span></span>
<span class="line"><span>func.func private @example_fn_attr() attributes {dialectName.attrName = false}</span></span></code></pre></div>`,5)),a(n,{type:"info",class:"source-link",text:"source"},{default:i(()=>e[14]||(e[14]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/13dc7a41f819f26358c64ee5e6f17d7be57713f5/src/mlir/Dialects/Func.jl#L145-L183",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",b,[t("summary",null,[e[16]||(e[16]=t("a",{id:"Reactant.MLIR.Dialects.func.return_-Tuple{Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.func.return_-Tuple{Vector{Reactant.MLIR.IR.Value}}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.func.return_")],-1)),e[17]||(e[17]=s()),a(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[19]||(e[19]=l(`<p><code>return_</code></p><p>The <code>func.return</code> operation represents a return operation within a function. The operation takes variable number of operands and produces no results. The operand number and types must match the signature of the function that contains the operation.</p><p><strong>Example</strong></p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>func.func @foo() -&gt; (i32, f8) {</span></span>
<span class="line"><span>  ...</span></span>
<span class="line"><span>  return %0, %1 : i32, f8</span></span>
<span class="line"><span>}</span></span></code></pre></div>`,4)),a(n,{type:"info",class:"source-link",text:"source"},{default:i(()=>e[18]||(e[18]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/13dc7a41f819f26358c64ee5e6f17d7be57713f5/src/mlir/Dialects/Func.jl#L219-L235",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})])])}const k=c(u,[["render",h]]);export{I as __pageData,k as default};
