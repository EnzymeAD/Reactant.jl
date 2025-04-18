import{_ as o,C as r,c,o as d,j as t,a as n,a2 as l,G as s,w as i}from"./chunks/framework.UgrWJckc.js";const A=JSON.parse('{"title":"Internal API","description":"","frontmatter":{},"headers":[],"relativePath":"api/internal.md","filePath":"api/internal.md","lastUpdated":null}'),p={name:"api/internal.md"},u={class:"jldocstring custom-block"},g={class:"jldocstring custom-block"},b={class:"jldocstring custom-block"},m={class:"jldocstring custom-block"};function f(h,e,_,k,y,j){const a=r("Badge");return d(),c("div",null,[e[16]||(e[16]=t("h1",{id:"Internal-API",tabindex:"-1"},[n("Internal API "),t("a",{class:"header-anchor",href:"#Internal-API","aria-label":'Permalink to "Internal API {#Internal-API}"'},"​")],-1)),e[17]||(e[17]=t("div",{class:"danger custom-block"},[t("p",{class:"custom-block-title"},"Private"),t("p",null,"These functions are not part of the public API and are subject to change at any time.")],-1)),t("details",u,[t("summary",null,[e[0]||(e[0]=t("a",{id:"Reactant.REDUB_ARGUMENTS_NAME",href:"#Reactant.REDUB_ARGUMENTS_NAME"},[t("span",{class:"jlbinding"},"Reactant.REDUB_ARGUMENTS_NAME")],-1)),e[1]||(e[1]=n()),s(a,{type:"info",class:"jlObjectType jlConstant",text:"Constant"})]),e[3]||(e[3]=l('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">REDUB_ARGUMENTS_NAME</span></span></code></pre></div><p>The variable name bound to <code>call_with_reactant</code>&#39;s tuple of arguments in its <code>@generated</code> method definition.</p><p>This binding can be used to manually reference/destructure <code>call_with_reactants</code> arguments</p><p>This is required because user arguments could have a name which clashes with whatever name we choose for our argument. Thus we gensym to create it.</p><p>This originates from <a href="https://github.com/JuliaLabs/Cassette.jl/blob/c29b237c1ec0deda3a1037ec519eebe216952bfe/src/overdub.jl#L154" target="_blank" rel="noreferrer">https://github.com/JuliaLabs/Cassette.jl/blob/c29b237c1ec0deda3a1037ec519eebe216952bfe/src/overdub.jl#L154</a></p>',5)),s(a,{type:"info",class:"source-link",text:"source"},{default:i(()=>e[2]||(e[2]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/aacf8c48a728b2242ce8c71633e2ea738ef1a45b/src/utils.jl#L17-L29",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",g,[t("summary",null,[e[4]||(e[4]=t("a",{id:"Reactant.Compiler.codegen_unflatten!",href:"#Reactant.Compiler.codegen_unflatten!"},[t("span",{class:"jlbinding"},"Reactant.Compiler.codegen_unflatten!")],-1)),e[5]||(e[5]=n()),s(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[7]||(e[7]=l('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">codegen_unflatten!</span></span></code></pre></div><p>Generate Julia code to wrap the XLA buffers back into the output result datatypes. The name is due to its similarity to the <code>unflatten</code> function in <code>jax.tree_util.register_pytree_node</code>.</p>',2)),s(a,{type:"info",class:"source-link",text:"source"},{default:i(()=>e[6]||(e[6]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/aacf8c48a728b2242ce8c71633e2ea738ef1a45b/src/Compiler.jl#L2159-L2164",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",b,[t("summary",null,[e[8]||(e[8]=t("a",{id:"Reactant.Compiler.codegen_flatten!",href:"#Reactant.Compiler.codegen_flatten!"},[t("span",{class:"jlbinding"},"Reactant.Compiler.codegen_flatten!")],-1)),e[9]||(e[9]=n()),s(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[11]||(e[11]=l('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">codegen_flatten!</span></span></code></pre></div><p>Generate Julia code to extract the XLA buffers from input arguments. The name is due to its similarity to the <code>flatten</code> function in <code>jax.tree_util.register_pytree_node</code>.</p><p><strong>Arguments</strong></p><ul><li><code>linear_args</code>: A list of arguments to be flattened.</li></ul><p><strong>Returns</strong></p><ul><li><p><code>flatten_names</code>: A list of <code>Symbol</code>s representing the names of the flattened arguments.</p></li><li><p><code>flatten_code</code>: A list of <code>Expr</code>s to extract the XLA buffers from the input arguments.</p></li></ul><p><strong>Note</strong></p><p>The <em>linearized arguments</em> do not directly refer to the are the arguments that have been flattened into a single list.</p>',8)),s(a,{type:"info",class:"source-link",text:"source"},{default:i(()=>e[10]||(e[10]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/aacf8c48a728b2242ce8c71633e2ea738ef1a45b/src/Compiler.jl#L1845-L1863",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",m,[t("summary",null,[e[12]||(e[12]=t("a",{id:"Reactant.Compiler.codegen_xla_call",href:"#Reactant.Compiler.codegen_xla_call"},[t("span",{class:"jlbinding"},"Reactant.Compiler.codegen_xla_call")],-1)),e[13]||(e[13]=n()),s(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[15]||(e[15]=l('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">codegen_xla_call</span></span></code></pre></div><p>Generate Julia code to call the XLA executable.</p><p><strong>Arguments</strong></p><ul><li><p><code>flatten_names</code>: A list of <code>Symbol</code>s representing the names of the flattened linear arguments.</p></li><li><p><code>nresults</code>: The number of results to expect.</p></li></ul>',4)),s(a,{type:"info",class:"source-link",text:"source"},{default:i(()=>e[14]||(e[14]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/aacf8c48a728b2242ce8c71633e2ea738ef1a45b/src/Compiler.jl#L2372-L2381",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})])])}const E=o(p,[["render",f]]);export{A as __pageData,E as default};
