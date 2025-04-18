import{_ as i,c as p,j as s,a as n,G as l,a2 as t,B as o,o as r}from"./chunks/framework.DXGduQyy.js";const q=JSON.parse('{"title":"LLVM Dialect","description":"","frontmatter":{},"headers":[],"relativePath":"api/llvm.md","filePath":"api/llvm.md","lastUpdated":null}'),c={name:"api/llvm.md"},d={class:"jldocstring custom-block"},m={class:"jldocstring custom-block"},u={class:"jldocstring custom-block"},b={class:"jldocstring custom-block"},v={class:"jldocstring custom-block"},f={class:"jldocstring custom-block"},g={class:"jldocstring custom-block"},h={class:"jldocstring custom-block"},L={class:"jldocstring custom-block"},y={class:"jldocstring custom-block"},R={class:"jldocstring custom-block"},M={class:"jldocstring custom-block"},I={class:"jldocstring custom-block"},k={class:"jldocstring custom-block"},j={class:"jldocstring custom-block"},D={class:"jldocstring custom-block"},T={class:"jldocstring custom-block"},x={class:"jldocstring custom-block"},V={class:"jldocstring custom-block"};function w(C,a,A,z,E,_){const e=o("Badge");return r(),p("div",null,[a[61]||(a[61]=s("h1",{id:"LLVM-Dialect",tabindex:"-1"},[n("LLVM Dialect "),s("a",{class:"header-anchor",href:"#LLVM-Dialect","aria-label":'Permalink to "LLVM Dialect {#LLVM-Dialect}"'},"​")],-1)),a[62]||(a[62]=s("p",null,[n("Refer to the "),s("a",{href:"https://mlir.llvm.org/docs/Dialects/LLVM/",target:"_blank",rel:"noreferrer"},"official documentation"),n(" for more details.")],-1)),s("details",d,[s("summary",null,[a[0]||(a[0]=s("a",{id:"Reactant.MLIR.Dialects.llvm.call-Tuple{Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.llvm.call-Tuple{Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.call")],-1)),a[1]||(a[1]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[2]||(a[2]=t(`<p><code>call</code></p><p>In LLVM IR, functions may return either 0 or 1 value. LLVM IR dialect implements this behavior by providing a variadic <code>call</code> operation for 0- and 1-result functions. Even though MLIR supports multi-result functions, LLVM IR dialect disallows them.</p><p>The <code>call</code> instruction supports both direct and indirect calls. Direct calls start with a function name (<code>@</code>-prefixed) and indirect calls start with an SSA value (<code>%</code>-prefixed). The direct callee, if present, is stored as a function attribute <code>callee</code>. For indirect calls, the callee is of <code>!llvm.ptr</code> type and is stored as the first value in <code>callee_operands</code>. If and only if the callee is a variadic function, the <code>var_callee_type</code> attribute must carry the variadic LLVM function type. The trailing type list contains the optional indirect callee type and the MLIR function type, which differs from the LLVM function type that uses an explicit void type to model functions that do not return a value.</p><p>Examples:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// Direct call without arguments and with one result.</span></span>
<span class="line"><span>%0 = llvm.call @foo() : () -&gt; (f32)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// Direct call with arguments and without a result.</span></span>
<span class="line"><span>llvm.call @bar(%0) : (f32) -&gt; ()</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// Indirect call with an argument and without a result.</span></span>
<span class="line"><span>%1 = llvm.mlir.addressof @foo : !llvm.ptr</span></span>
<span class="line"><span>llvm.call %1(%0) : !llvm.ptr, (f32) -&gt; ()</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// Direct variadic call.</span></span>
<span class="line"><span>llvm.call @printf(%0, %1) vararg(!llvm.func&lt;i32 (ptr, ...)&gt;) : (!llvm.ptr, i32) -&gt; i32</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// Indirect variadic call</span></span>
<span class="line"><span>llvm.call %1(%0) vararg(!llvm.func&lt;void (...)&gt;) : !llvm.ptr, (i32) -&gt; ()</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L362-L400" target="_blank" rel="noreferrer">source</a></p>`,6))]),s("details",m,[s("summary",null,[a[3]||(a[3]=s("a",{id:"Reactant.MLIR.Dialects.llvm.call_intrinsic-Tuple{Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.llvm.call_intrinsic-Tuple{Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.call_intrinsic")],-1)),a[4]||(a[4]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[5]||(a[5]=s("p",null,[s("code",null,"call_intrinsic")],-1)),a[6]||(a[6]=s("p",null,"Call the specified llvm intrinsic. If the intrinsic is overloaded, use the MLIR function type of this op to determine which intrinsic to call.",-1)),a[7]||(a[7]=s("p",null,[s("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L320-L325",target:"_blank",rel:"noreferrer"},"source")],-1))]),s("details",u,[s("summary",null,[a[8]||(a[8]=s("a",{id:"Reactant.MLIR.Dialects.llvm.comdat-Tuple{}",href:"#Reactant.MLIR.Dialects.llvm.comdat-Tuple{}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.comdat")],-1)),a[9]||(a[9]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[10]||(a[10]=t(`<p><code>comdat</code></p><p>Provides access to object file COMDAT section/group functionality.</p><p>Examples:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>llvm.comdat @__llvm_comdat {</span></span>
<span class="line"><span>  llvm.comdat_selector @any any</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L470-L482" target="_blank" rel="noreferrer">source</a></p>`,5))]),s("details",b,[s("summary",null,[a[11]||(a[11]=s("a",{id:"Reactant.MLIR.Dialects.llvm.comdat_selector-Tuple{}",href:"#Reactant.MLIR.Dialects.llvm.comdat_selector-Tuple{}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.comdat_selector")],-1)),a[12]||(a[12]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[13]||(a[13]=t(`<p><code>comdat_selector</code></p><p>Provides access to object file COMDAT section/group functionality.</p><p>Examples:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>llvm.comdat @__llvm_comdat {</span></span>
<span class="line"><span>  llvm.comdat_selector @any any</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L502-L514" target="_blank" rel="noreferrer">source</a></p>`,5))]),s("details",v,[s("summary",null,[a[14]||(a[14]=s("a",{id:"Reactant.MLIR.Dialects.llvm.func-Tuple{}",href:"#Reactant.MLIR.Dialects.llvm.func-Tuple{}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.func")],-1)),a[15]||(a[15]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[16]||(a[16]=t(`<p><code>func</code></p><p>MLIR functions are defined by an operation that is not built into the IR itself. The LLVM dialect provides an <code>llvm.func</code> operation to define functions compatible with LLVM IR. These functions have LLVM dialect function type but use MLIR syntax to express it. They are required to have exactly one result type. LLVM function operation is intended to capture additional properties of LLVM functions, such as linkage and calling convention, that may be modeled differently by the built-in MLIR function.</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// The type of @bar is !llvm&lt;&quot;i64 (i64)&quot;&gt;</span></span>
<span class="line"><span>llvm.func @bar(%arg0: i64) -&gt; i64 {</span></span>
<span class="line"><span>  llvm.return %arg0 : i64</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// Type type of @foo is !llvm&lt;&quot;void (i64)&quot;&gt;</span></span>
<span class="line"><span>// !llvm.void type is omitted</span></span>
<span class="line"><span>llvm.func @foo(%arg0: i64) {</span></span>
<span class="line"><span>  llvm.return</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// A function with \`internal\` linkage.</span></span>
<span class="line"><span>llvm.func internal @internal_func() {</span></span>
<span class="line"><span>  llvm.return</span></span>
<span class="line"><span>}</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L1487-L1515" target="_blank" rel="noreferrer">source</a></p>`,4))]),s("details",f,[s("summary",null,[a[17]||(a[17]=s("a",{id:"Reactant.MLIR.Dialects.llvm.getelementptr-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.llvm.getelementptr-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.getelementptr")],-1)),a[18]||(a[18]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[19]||(a[19]=t(`<p><code>getelementptr</code></p><p>This operation mirrors LLVM IRs &#39;getelementptr&#39; operation that is used to perform pointer arithmetic.</p><p>Like in LLVM IR, it is possible to use both constants as well as SSA values as indices. In the case of indexing within a structure, it is required to either use constant indices directly, or supply a constant SSA value.</p><p>An optional &#39;inbounds&#39; attribute specifies the low-level pointer arithmetic overflow behavior that LLVM uses after lowering the operation to LLVM IR.</p><p>Examples:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// GEP with an SSA value offset</span></span>
<span class="line"><span>%0 = llvm.getelementptr %1[%2] : (!llvm.ptr, i64) -&gt; !llvm.ptr, f32</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// GEP with a constant offset and the inbounds attribute set</span></span>
<span class="line"><span>%0 = llvm.getelementptr inbounds %1[3] : (!llvm.ptr) -&gt; !llvm.ptr, f32</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// GEP with constant offsets into a structure</span></span>
<span class="line"><span>%0 = llvm.getelementptr %1[0, 1]</span></span>
<span class="line"><span>   : (!llvm.ptr) -&gt; !llvm.ptr, !llvm.struct&lt;(i32, f32)&gt;</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L990-L1016" target="_blank" rel="noreferrer">source</a></p>`,7))]),s("details",g,[s("summary",null,[a[20]||(a[20]=s("a",{id:"Reactant.MLIR.Dialects.llvm.inline_asm-Tuple{Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.llvm.inline_asm-Tuple{Vector{Reactant.MLIR.IR.Value}}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.inline_asm")],-1)),a[21]||(a[21]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[22]||(a[22]=s("p",null,[s("code",null,"inline_asm")],-1)),a[23]||(a[23]=s("p",null,[n("The InlineAsmOp mirrors the underlying LLVM semantics with a notable exception: the embedded "),s("code",null,"asm_string"),n(" is not allowed to define or reference any symbol or any global variable: only the operands of the op may be read, written, or referenced. Attempting to define or reference any symbol or any global behavior is considered undefined behavior at this time.")],-1)),a[24]||(a[24]=s("p",null,[s("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L1310-L1319",target:"_blank",rel:"noreferrer"},"source")],-1))]),s("details",h,[s("summary",null,[a[25]||(a[25]=s("a",{id:"Reactant.MLIR.Dialects.llvm.linker_options-Tuple{}",href:"#Reactant.MLIR.Dialects.llvm.linker_options-Tuple{}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.linker_options")],-1)),a[26]||(a[26]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[27]||(a[27]=t(`<p><code>linker_options</code></p><p>Pass the given options to the linker when the resulting object file is linked. This is used extensively on Windows to determine the C runtime that the object files should link against.</p><p>Examples:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// Link against the MSVC static threaded CRT.</span></span>
<span class="line"><span>llvm.linker_options [&quot;/DEFAULTLIB:&quot;, &quot;libcmt&quot;]</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// Link against aarch64 compiler-rt builtins</span></span>
<span class="line"><span>llvm.linker_options [&quot;-l&quot;, &quot;clang_rt.builtins-aarch64&quot;]</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L1715-L1730" target="_blank" rel="noreferrer">source</a></p>`,5))]),s("details",L,[s("summary",null,[a[28]||(a[28]=s("a",{id:"Reactant.MLIR.Dialects.llvm.load-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.llvm.load-Tuple{Reactant.MLIR.IR.Value}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.load")],-1)),a[29]||(a[29]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[30]||(a[30]=t(`<p><code>load</code></p><p>The <code>load</code> operation is used to read from memory. A load may be marked as atomic, volatile, and/or nontemporal, and takes a number of optional attributes that specify aliasing information.</p><p>An atomic load only supports a limited set of pointer, integer, and floating point types, and requires an explicit alignment.</p><p>Examples:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// A volatile load of a float variable.</span></span>
<span class="line"><span>%0 = llvm.load volatile %ptr : !llvm.ptr -&gt; f32</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// A nontemporal load of a float variable.</span></span>
<span class="line"><span>%0 = llvm.load %ptr {nontemporal} : !llvm.ptr -&gt; f32</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// An atomic load of an integer variable.</span></span>
<span class="line"><span>%0 = llvm.load %ptr atomic monotonic {alignment = 8 : i64}</span></span>
<span class="line"><span>    : !llvm.ptr -&gt; i64</span></span></code></pre></div><p>See the following link for more details: <a href="https://llvm.org/docs/LangRef.html#load-instruction" target="_blank" rel="noreferrer">https://llvm.org/docs/LangRef.html#load-instruction</a></p><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L1750-L1775" target="_blank" rel="noreferrer">source</a></p>`,7))]),s("details",y,[s("summary",null,[a[31]||(a[31]=s("a",{id:"Reactant.MLIR.Dialects.llvm.mlir_addressof-Tuple{}",href:"#Reactant.MLIR.Dialects.llvm.mlir_addressof-Tuple{}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.mlir_addressof")],-1)),a[32]||(a[32]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[33]||(a[33]=t(`<p><code>mlir_addressof</code></p><p>Creates an SSA value containing a pointer to a global variable or constant defined by <code>llvm.mlir.global</code>. The global value can be defined after its first referenced. If the global value is a constant, storing into it is not allowed.</p><p>Examples:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>func @foo() {</span></span>
<span class="line"><span>  // Get the address of a global variable.</span></span>
<span class="line"><span>  %0 = llvm.mlir.addressof @const : !llvm.ptr</span></span>
<span class="line"><span></span></span>
<span class="line"><span>  // Use it as a regular pointer.</span></span>
<span class="line"><span>  %1 = llvm.load %0 : !llvm.ptr -&gt; i32</span></span>
<span class="line"><span></span></span>
<span class="line"><span>  // Get the address of a function.</span></span>
<span class="line"><span>  %2 = llvm.mlir.addressof @foo : !llvm.ptr</span></span>
<span class="line"><span></span></span>
<span class="line"><span>  // The function address can be used for indirect calls.</span></span>
<span class="line"><span>  llvm.call %2() : !llvm.ptr, () -&gt; ()</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// Define the global.</span></span>
<span class="line"><span>llvm.mlir.global @const(42 : i32) : i32</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L84-L112" target="_blank" rel="noreferrer">source</a></p>`,5))]),s("details",R,[s("summary",null,[a[34]||(a[34]=s("a",{id:"Reactant.MLIR.Dialects.llvm.mlir_constant-Tuple{}",href:"#Reactant.MLIR.Dialects.llvm.mlir_constant-Tuple{}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.mlir_constant")],-1)),a[35]||(a[35]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[36]||(a[36]=t(`<p><code>mlir_constant</code></p><p>Unlike LLVM IR, MLIR does not have first-class constant values. Therefore, all constants must be created as SSA values before being used in other operations. <code>llvm.mlir.constant</code> creates such values for scalars, vectors, strings, and structs. It has a mandatory <code>value</code> attribute whose type depends on the type of the constant value. The type of the constant value must correspond to the attribute type converted to LLVM IR type.</p><p>When creating constant scalars, the <code>value</code> attribute must be either an integer attribute or a floating point attribute. The type of the attribute may be omitted for <code>i64</code> and <code>f64</code> types that are implied.</p><p>When creating constant vectors, the <code>value</code> attribute must be either an array attribute, a dense attribute, or a sparse attribute that contains integers or floats. The number of elements in the result vector must match the number of elements in the attribute.</p><p>When creating constant strings, the <code>value</code> attribute must be a string attribute. The type of the constant must be an LLVM array of <code>i8</code>s, and the length of the array must match the length of the attribute.</p><p>When creating constant structs, the <code>value</code> attribute must be an array attribute that contains integers or floats. The type of the constant must be an LLVM struct type. The number of fields in the struct must match the number of elements in the attribute, and the type of each LLVM struct field must correspond to the type of the corresponding attribute element converted to LLVM IR.</p><p>Examples:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// Integer constant, internal i32 is mandatory</span></span>
<span class="line"><span>%0 = llvm.mlir.constant(42 : i32) : i32</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// It&#39;s okay to omit i64.</span></span>
<span class="line"><span>%1 = llvm.mlir.constant(42) : i64</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// Floating point constant.</span></span>
<span class="line"><span>%2 = llvm.mlir.constant(42.0 : f32) : f32</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// Splat dense vector constant.</span></span>
<span class="line"><span>%3 = llvm.mlir.constant(dense&lt;1.0&gt; : vector&lt;4xf32&gt;) : vector&lt;4xf32&gt;</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L572-L617" target="_blank" rel="noreferrer">source</a></p>`,9))]),s("details",M,[s("summary",null,[a[37]||(a[37]=s("a",{id:"Reactant.MLIR.Dialects.llvm.mlir_global-Tuple{}",href:"#Reactant.MLIR.Dialects.llvm.mlir_global-Tuple{}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.mlir_global")],-1)),a[38]||(a[38]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[39]||(a[39]=t(`<p><code>mlir_global</code></p><p>Since MLIR allows for arbitrary operations to be present at the top level, global variables are defined using the <code>llvm.mlir.global</code> operation. Both global constants and variables can be defined, and the value may also be initialized in both cases.</p><p>There are two forms of initialization syntax. Simple constants that can be represented as MLIR attributes can be given in-line:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>llvm.mlir.global @variable(32.0 : f32) : f32</span></span></code></pre></div><p>This initialization and type syntax is similar to <code>llvm.mlir.constant</code> and may use two types: one for MLIR attribute and another for the LLVM value. These types must be compatible.</p><p>More complex constants that cannot be represented as MLIR attributes can be given in an initializer region:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// This global is initialized with the equivalent of:</span></span>
<span class="line"><span>//   i32* getelementptr (i32* @g2, i32 2)</span></span>
<span class="line"><span>llvm.mlir.global constant @int_gep() : !llvm.ptr {</span></span>
<span class="line"><span>  %0 = llvm.mlir.addressof @g2 : !llvm.ptr</span></span>
<span class="line"><span>  %1 = llvm.mlir.constant(2 : i32) : i32</span></span>
<span class="line"><span>  %2 = llvm.getelementptr %0[%1]</span></span>
<span class="line"><span>     : (!llvm.ptr, i32) -&gt; !llvm.ptr, i32</span></span>
<span class="line"><span>  // The initializer region must end with \`llvm.return\`.</span></span>
<span class="line"><span>  llvm.return %2 : !llvm.ptr</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>Only one of the initializer attribute or initializer region may be provided.</p><p><code>llvm.mlir.global</code> must appear at top-level of the enclosing module. It uses an @-identifier for its value, which will be uniqued by the module with respect to other @-identifiers in it.</p><p>Examples:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// Global values use @-identifiers.</span></span>
<span class="line"><span>llvm.mlir.global constant @cst(42 : i32) : i32</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// Non-constant values must also be initialized.</span></span>
<span class="line"><span>llvm.mlir.global @variable(32.0 : f32) : f32</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// Strings are expected to be of wrapped LLVM i8 array type and do not</span></span>
<span class="line"><span>// automatically include the trailing zero.</span></span>
<span class="line"><span>llvm.mlir.global @string(&quot;abc&quot;) : !llvm.array&lt;3 x i8&gt;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// For strings globals, the trailing type may be omitted.</span></span>
<span class="line"><span>llvm.mlir.global constant @no_trailing_type(&quot;foo bar&quot;)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// A complex initializer is constructed with an initializer region.</span></span>
<span class="line"><span>llvm.mlir.global constant @int_gep() : !llvm.ptr {</span></span>
<span class="line"><span>  %0 = llvm.mlir.addressof @g2 : !llvm.ptr</span></span>
<span class="line"><span>  %1 = llvm.mlir.constant(2 : i32) : i32</span></span>
<span class="line"><span>  %2 = llvm.getelementptr %0[%1]</span></span>
<span class="line"><span>     : (!llvm.ptr, i32) -&gt; !llvm.ptr, i32</span></span>
<span class="line"><span>  llvm.return %2 : !llvm.ptr</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>Similarly to functions, globals have a linkage attribute. In the custom syntax, this attribute is placed between <code>llvm.mlir.global</code> and the optional <code>constant</code> keyword. If the attribute is omitted, <code>external</code> linkage is assumed by default.</p><p>Examples:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// A constant with internal linkage will not participate in linking.</span></span>
<span class="line"><span>llvm.mlir.global internal constant @cst(42 : i32) : i32</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// By default, &quot;external&quot; linkage is assumed and the global participates in</span></span>
<span class="line"><span>// symbol resolution at link-time.</span></span>
<span class="line"><span>llvm.mlir.global @glob(0 : f32) : f32</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// Alignment is optional</span></span>
<span class="line"><span>llvm.mlir.global private constant @y(dense&lt;1.0&gt; : tensor&lt;8xf32&gt;) : !llvm.array&lt;8 x f32&gt;</span></span></code></pre></div><p>Like global variables in LLVM IR, globals can have an (optional) alignment attribute using keyword <code>alignment</code>. The integer value of the alignment must be a positive integer that is a power of 2.</p><p>Examples:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// Alignment is optional</span></span>
<span class="line"><span>llvm.mlir.global private constant @y(dense&lt;1.0&gt; : tensor&lt;8xf32&gt;) { alignment = 32 : i64 } : !llvm.array&lt;8 x f32&gt;</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L1131-L1227" target="_blank" rel="noreferrer">source</a></p>`,18))]),s("details",I,[s("summary",null,[a[40]||(a[40]=s("a",{id:"Reactant.MLIR.Dialects.llvm.mlir_global_ctors-Tuple{}",href:"#Reactant.MLIR.Dialects.llvm.mlir_global_ctors-Tuple{}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.mlir_global_ctors")],-1)),a[41]||(a[41]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[42]||(a[42]=t(`<p><code>mlir_global_ctors</code></p><p>Specifies a list of constructor functions and priorities. The functions referenced by this array will be called in ascending order of priority (i.e. lowest first) when the module is loaded. The order of functions with the same priority is not defined. This operation is translated to LLVM&#39;s global_ctors global variable. The initializer functions are run at load time. The <code>data</code> field present in LLVM&#39;s global_ctors variable is not modeled here.</p><p>Examples:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>llvm.mlir.global_ctors {@ctor}</span></span>
<span class="line"><span></span></span>
<span class="line"><span>llvm.func @ctor() {</span></span>
<span class="line"><span>  ...</span></span>
<span class="line"><span>  llvm.return</span></span>
<span class="line"><span>}</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L1048-L1069" target="_blank" rel="noreferrer">source</a></p>`,5))]),s("details",k,[s("summary",null,[a[43]||(a[43]=s("a",{id:"Reactant.MLIR.Dialects.llvm.mlir_global_dtors-Tuple{}",href:"#Reactant.MLIR.Dialects.llvm.mlir_global_dtors-Tuple{}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.mlir_global_dtors")],-1)),a[44]||(a[44]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[45]||(a[45]=t(`<p><code>mlir_global_dtors</code></p><p>Specifies a list of destructor functions and priorities. The functions referenced by this array will be called in descending order of priority (i.e. highest first) when the module is unloaded. The order of functions with the same priority is not defined. This operation is translated to LLVM&#39;s global_dtors global variable. The <code>data</code> field present in LLVM&#39;s global_dtors variable is not modeled here.</p><p>Examples:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>llvm.func @dtor() {</span></span>
<span class="line"><span>  llvm.return</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>llvm.mlir.global_dtors {@dtor}</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L1091-L1109" target="_blank" rel="noreferrer">source</a></p>`,5))]),s("details",j,[s("summary",null,[a[46]||(a[46]=s("a",{id:"Reactant.MLIR.Dialects.llvm.mlir_none-Tuple{}",href:"#Reactant.MLIR.Dialects.llvm.mlir_none-Tuple{}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.mlir_none")],-1)),a[47]||(a[47]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[48]||(a[48]=t('<p><code>mlir_none</code></p><p>Unlike LLVM IR, MLIR does not have first-class token values. They must be explicitly created as SSA values using <code>llvm.mlir.none</code>. This operation has no operands or attributes, and returns a none token value of a wrapped LLVM IR pointer type.</p><p>Examples:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>%0 = llvm.mlir.none : !llvm.token</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L1847-L1860" target="_blank" rel="noreferrer">source</a></p>',5))]),s("details",D,[s("summary",null,[a[49]||(a[49]=s("a",{id:"Reactant.MLIR.Dialects.llvm.mlir_poison-Tuple{}",href:"#Reactant.MLIR.Dialects.llvm.mlir_poison-Tuple{}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.mlir_poison")],-1)),a[50]||(a[50]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[51]||(a[51]=t(`<p><code>mlir_poison</code></p><p>Unlike LLVM IR, MLIR does not have first-class poison values. Such values must be created as SSA values using <code>llvm.mlir.poison</code>. This operation has no operands or attributes. It creates a poison value of the specified LLVM IR dialect type.</p><p><strong>Example</strong></p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// Create a poison value for a structure with a 32-bit integer followed</span></span>
<span class="line"><span>// by a float.</span></span>
<span class="line"><span>%0 = llvm.mlir.poison : !llvm.struct&lt;(i32, f32)&gt;</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L1908-L1923" target="_blank" rel="noreferrer">source</a></p>`,5))]),s("details",T,[s("summary",null,[a[52]||(a[52]=s("a",{id:"Reactant.MLIR.Dialects.llvm.mlir_undef-Tuple{}",href:"#Reactant.MLIR.Dialects.llvm.mlir_undef-Tuple{}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.mlir_undef")],-1)),a[53]||(a[53]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[54]||(a[54]=t(`<p><code>mlir_undef</code></p><p>Unlike LLVM IR, MLIR does not have first-class undefined values. Such values must be created as SSA values using <code>llvm.mlir.undef</code>. This operation has no operands or attributes. It creates an undefined value of the specified LLVM IR dialect type.</p><p><strong>Example</strong></p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// Create a structure with a 32-bit integer followed by a float.</span></span>
<span class="line"><span>%0 = llvm.mlir.undef : !llvm.struct&lt;(i32, f32)&gt;</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L2378-L2392" target="_blank" rel="noreferrer">source</a></p>`,5))]),s("details",x,[s("summary",null,[a[55]||(a[55]=s("a",{id:"Reactant.MLIR.Dialects.llvm.mlir_zero-Tuple{}",href:"#Reactant.MLIR.Dialects.llvm.mlir_zero-Tuple{}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.mlir_zero")],-1)),a[56]||(a[56]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[57]||(a[57]=t(`<p><code>mlir_zero</code></p><p>Unlike LLVM IR, MLIR does not have first-class zero-initialized values. Such values must be created as SSA values using <code>llvm.mlir.zero</code>. This operation has no operands or attributes. It creates a zero-initialized value of the specified LLVM IR dialect type.</p><p><strong>Example</strong></p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// Create a zero-initialized value for a structure with a 32-bit integer</span></span>
<span class="line"><span>// followed by a float.</span></span>
<span class="line"><span>%0 = llvm.mlir.zero : !llvm.struct&lt;(i32, f32)&gt;</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L2492-L2507" target="_blank" rel="noreferrer">source</a></p>`,5))]),s("details",V,[s("summary",null,[a[58]||(a[58]=s("a",{id:"Reactant.MLIR.Dialects.llvm.store-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.llvm.store-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[s("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.llvm.store")],-1)),a[59]||(a[59]=n()),l(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),a[60]||(a[60]=t(`<p><code>store</code></p><p>The <code>store</code> operation is used to write to memory. A store may be marked as atomic, volatile, and/or nontemporal, and takes a number of optional attributes that specify aliasing information.</p><p>An atomic store only supports a limited set of pointer, integer, and floating point types, and requires an explicit alignment.</p><p>Examples:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// A volatile store of a float variable.</span></span>
<span class="line"><span>llvm.store volatile %val, %ptr : f32, !llvm.ptr</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// A nontemporal store of a float variable.</span></span>
<span class="line"><span>llvm.store %val, %ptr {nontemporal} : f32, !llvm.ptr</span></span>
<span class="line"><span></span></span>
<span class="line"><span>// An atomic store of an integer variable.</span></span>
<span class="line"><span>llvm.store %val, %ptr atomic monotonic {alignment = 8 : i64}</span></span>
<span class="line"><span>    : i64, !llvm.ptr</span></span></code></pre></div><p>See the following link for more details: <a href="https://llvm.org/docs/LangRef.html#store-instruction" target="_blank" rel="noreferrer">https://llvm.org/docs/LangRef.html#store-instruction</a></p><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/f917bfeb48c3b53ff892bd55a5194ab98391483b/src/mlir/Dialects/Llvm.jl#L2158-L2183" target="_blank" rel="noreferrer">source</a></p>`,7))])])}const O=i(c,[["render",w]]);export{q as __pageData,O as default};
