import{_ as l,c as p,j as i,a,G as e,a2 as n,B as h,o as k}from"./chunks/framework.C0lb17Xd.js";const v=JSON.parse('{"title":"Core Reactant API","description":"","frontmatter":{},"headers":[],"relativePath":"api/api.md","filePath":"api/api.md","lastUpdated":null}'),r={name:"api/api.md"},o={class:"jldocstring custom-block"},d={class:"jldocstring custom-block"},c={class:"jldocstring custom-block"},g={class:"jldocstring custom-block"},y={class:"jldocstring custom-block"},E={class:"jldocstring custom-block"},F={class:"jldocstring custom-block"};function u(C,s,f,b,A,m){const t=h("Badge");return k(),p("div",null,[s[21]||(s[21]=i("h1",{id:"Core-Reactant-API",tabindex:"-1"},[a("Core Reactant API "),i("a",{class:"header-anchor",href:"#Core-Reactant-API","aria-label":'Permalink to "Core Reactant API {#Core-Reactant-API}"'},"​")],-1)),s[22]||(s[22]=i("h2",{id:"Compile-API",tabindex:"-1"},[a("Compile API "),i("a",{class:"header-anchor",href:"#Compile-API","aria-label":'Permalink to "Compile API {#Compile-API}"'},"​")],-1)),i("details",o,[i("summary",null,[s[0]||(s[0]=i("a",{id:"Reactant.Compiler.@compile",href:"#Reactant.Compiler.@compile"},[i("span",{class:"jlbinding"},"Reactant.Compiler.@compile")],-1)),s[1]||(s[1]=a()),e(t,{type:"info",class:"jlObjectType jlMacro",text:"Macro"})]),s[2]||(s[2]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@compile</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [optimize </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> ...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] [no_nan </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] [sync </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">f</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/2afb386695721c46450ca6bfdc539c47eea54f5d/src/Compiler.jl#L609-L611" target="_blank" rel="noreferrer">source</a></p>',2))]),i("details",d,[i("summary",null,[s[3]||(s[3]=i("a",{id:"Reactant.Compiler.@jit",href:"#Reactant.Compiler.@jit"},[i("span",{class:"jlbinding"},"Reactant.Compiler.@jit")],-1)),s[4]||(s[4]=a()),e(t,{type:"info",class:"jlObjectType jlMacro",text:"Macro"})]),s[5]||(s[5]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@jit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [optimize </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> ...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] [no_nan </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] [sync </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">f</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Run @compile f(args..) then immediately execute it</p><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/2afb386695721c46450ca6bfdc539c47eea54f5d/src/Compiler.jl#L617-L621" target="_blank" rel="noreferrer">source</a></p>',3))]),s[23]||(s[23]=i("h2",{id:"ReactantCore-API",tabindex:"-1"},[a("ReactantCore API "),i("a",{class:"header-anchor",href:"#ReactantCore-API","aria-label":'Permalink to "ReactantCore API {#ReactantCore-API}"'},"​")],-1)),i("details",c,[i("summary",null,[s[6]||(s[6]=i("a",{id:"ReactantCore.@trace",href:"#ReactantCore.@trace"},[i("span",{class:"jlbinding"},"ReactantCore.@trace")],-1)),s[7]||(s[7]=a()),e(t,{type:"info",class:"jlObjectType jlMacro",text:"Macro"})]),s[8]||(s[8]=n(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@trace</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">expr</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span></span></code></pre></div><p>Converts certain expressions like control flow into a Reactant friendly form. Importantly, if no traced value is found inside the expression, then there is no overhead.</p><p><strong>Currently Supported</strong></p><ul><li><p><code>if</code> conditions (with <code>elseif</code> and other niceties) (<code>@trace if ...</code>)</p></li><li><p><code>if</code> statements with a preceeding assignment (<code>@trace a = if ...</code>) (note the positioning of the macro needs to be before the assignment and not before the <code>if</code>)</p></li><li><p><code>for</code> statements with a single induction variable iterating over a syntactic <code>StepRange</code> of integers.</p></li></ul><p><strong>Special Considerations</strong></p><ul><li>Apply <code>@trace</code> only at the outermost <code>if</code>. Nested <code>if</code> statements will be automatically expanded into the correct form.</li></ul><p><strong>Extended Help</strong></p><p><strong>Caveats (Deviations from Core Julia Semantics)</strong></p><p><strong>New variables introduced</strong></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@trace</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    p </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">else</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>In the outer scope <code>p</code> is not defined if <code>x ≤ 0</code>. However, for the traced version, it is defined and set to a dummy value.</p><p><strong>Short Circuiting Operations</strong></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@trace</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> z </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">else</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p><code>&amp;&amp;</code> and <code>||</code> are short circuiting operations. In the traced version, we replace them with <code>&amp;</code> and <code>|</code> respectively.</p><p><strong>Type-Unstable Branches</strong></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@trace</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0f0</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">else</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>This will not compile since <code>y</code> is a <code>Float32</code> in one branch and a <code>Float64</code> in the other. You need to ensure that all branches have the same type.</p><p>Another example is the following for loop which changes the type of <code>x</code> between iterations.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> ...</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> # ConcreteRArray{Int64, 1}</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> i </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1f0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5f0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10f0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> i </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># ConcreteRArray{Float32, 1}</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p><strong>Certain Symbols are Reserved</strong></p><p>Symbols like [😦😃, :nothing, :missing, :Inf, :Inf16, :Inf32, :Inf64, :Base, :Core] are not allowed as variables in <code>@trace</code> expressions. While certain cases might work but these are not guaranteed to work. For example, the following will not work:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> fn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    nothing</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @trace</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    else</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2.0</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nothing</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/2afb386695721c46450ca6bfdc539c47eea54f5d/lib/ReactantCore/src/ReactantCore.jl#L25-L112" target="_blank" rel="noreferrer">source</a></p>`,23))]),s[24]||(s[24]=i("h2",{id:"Inspect-Generated-HLO",tabindex:"-1"},[a("Inspect Generated HLO "),i("a",{class:"header-anchor",href:"#Inspect-Generated-HLO","aria-label":'Permalink to "Inspect Generated HLO {#Inspect-Generated-HLO}"'},"​")],-1)),i("details",g,[i("summary",null,[s[9]||(s[9]=i("a",{id:"Reactant.Compiler.@code_hlo",href:"#Reactant.Compiler.@code_hlo"},[i("span",{class:"jlbinding"},"Reactant.Compiler.@code_hlo")],-1)),s[10]||(s[10]=a()),e(t,{type:"info",class:"jlObjectType jlMacro",text:"Macro"})]),s[11]||(s[11]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@code_hlo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [optimize </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> ...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] [no_nan </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> &lt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">f</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/2afb386695721c46450ca6bfdc539c47eea54f5d/src/Compiler.jl#L597-L599" target="_blank" rel="noreferrer">source</a></p>',2))]),s[25]||(s[25]=i("h2",{id:"Profile-XLA",tabindex:"-1"},[a("Profile XLA "),i("a",{class:"header-anchor",href:"#Profile-XLA","aria-label":'Permalink to "Profile XLA {#Profile-XLA}"'},"​")],-1)),s[26]||(s[26]=i("p",null,[a("Reactant can hook into XLA's profiler to generate compilation and execution traces. See the "),i("a",{href:"/Reactant.jl/previews/PR535/tutorials/profiling#profiling"},"profiling tutorial"),a(" for more details.")],-1)),i("details",y,[i("summary",null,[s[12]||(s[12]=i("a",{id:"Reactant.Profiler.with_profiler",href:"#Reactant.Profiler.with_profiler"},[i("span",{class:"jlbinding"},"Reactant.Profiler.with_profiler")],-1)),s[13]||(s[13]=a()),e(t,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[14]||(s[14]=n(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">with_profiler</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(f, trace_output_dir</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">String</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; trace_device</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, trace_host</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, create_perfetto_link</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Runs the provided function under a profiler for XLA (similar to <a href="https://jax.readthedocs.io/en/latest/profiling.html" target="_blank" rel="noreferrer">JAX&#39;s profiler</a>). The traces will be exported in the provided folder and can be seen using tools like <a href="https://ui.perfetto.dev" target="_blank" rel="noreferrer">perfetto.dev</a>. It will return the return values from the function. The <code>create_perfetto_link</code> parameter can be used to automatically generate a perfetto url to visualize the trace.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">with_profiler</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;./traces/&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    compiled_func </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> myfunc</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y, z)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    compiled_func</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y, z)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/2afb386695721c46450ca6bfdc539c47eea54f5d/src/Profiler.jl#L6-L21" target="_blank" rel="noreferrer">source</a></p>`,4))]),i("details",E,[i("summary",null,[s[15]||(s[15]=i("a",{id:"Reactant.Profiler.annotate",href:"#Reactant.Profiler.annotate"},[i("span",{class:"jlbinding"},"Reactant.Profiler.annotate")],-1)),s[16]||(s[16]=a()),e(t,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),s[17]||(s[17]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">annotate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(f, name, [level</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">TRACE_ME_LEVEL_CRITICAL])</span></span></code></pre></div><p>Generate an annotation in the current trace.</p><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/2afb386695721c46450ca6bfdc539c47eea54f5d/src/Profiler.jl#L61-L65" target="_blank" rel="noreferrer">source</a></p>',3))]),i("details",F,[i("summary",null,[s[18]||(s[18]=i("a",{id:"Reactant.Profiler.@annotate",href:"#Reactant.Profiler.@annotate"},[i("span",{class:"jlbinding"},"Reactant.Profiler.@annotate")],-1)),s[19]||(s[19]=a()),e(t,{type:"info",class:"jlObjectType jlMacro",text:"Macro"})]),s[20]||(s[20]=n(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@annotate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [name] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> foo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(a, b, c)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    ...</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>The created function will generate an annotation in the captured XLA profiles.</p><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/2afb386695721c46450ca6bfdc539c47eea54f5d/src/Profiler.jl#L77-L83" target="_blank" rel="noreferrer">source</a></p>`,3))])])}const _=l(r,[["render",u]]);export{v as __pageData,_ as default};
