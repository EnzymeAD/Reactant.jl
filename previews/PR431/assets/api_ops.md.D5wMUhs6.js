import{_ as l,c as p,j as i,a,G as n,a2 as e,B as h,o as r}from"./chunks/framework.DjkImPyO.js";const b=JSON.parse('{"title":"Reactant.Ops API","description":"","frontmatter":{},"headers":[],"relativePath":"api/ops.md","filePath":"api/ops.md","lastUpdated":null}'),k={name:"api/ops.md"},d={class:"jldocstring custom-block"},o={class:"jldocstring custom-block"},E={class:"jldocstring custom-block"},g={class:"jldocstring custom-block"};function y(c,s,u,F,C,m){const t=h("Badge");return r(),p("div",null,[s[12]||(s[12]=i("h1",{id:"Reactant.Ops-API",tabindex:"-1"},[i("code",null,"Reactant.Ops"),a(" API "),i("a",{class:"header-anchor",href:"#Reactant.Ops-API","aria-label":'Permalink to "`Reactant.Ops` API {#Reactant.Ops-API}"'},"​")],-1)),s[13]||(s[13]=i("p",null,[i("code",null,"Reactant.Ops"),a(" module provides a high-level API to construct MLIR operations without having to directly interact with the different dialects.")],-1)),s[14]||(s[14]=i("p",null,[a("Currently we haven't documented all the functions in "),i("code",null,"Reactant.Ops"),a(".")],-1)),i("details",d,[i("summary",null,[s[0]||(s[0]=i("a",{id:"Reactant.Ops.hlo_call-Tuple{Any, Vararg{Any}}",href:"#Reactant.Ops.hlo_call-Tuple{Any, Vararg{Any}}"},[i("span",{class:"jlbinding"},"Reactant.Ops.hlo_call")],-1)),s[1]||(s[1]=a()),n(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[2]||(s[2]=e(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Ops</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">hlo_call</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(mlir_code</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">String</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Vararg{AnyTracedRArray}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">...; func_name</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">String</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;main&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> NTuple{N, AnyTracedRArray}</span></span></code></pre></div><p>Given a MLIR module given as a string, calls the function identified by the <code>func_name</code> keyword parameter (default &quot;main&quot;) with the provided arguments and return a tuple for each result of the call.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@jit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">          Ops</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">hlo_call</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">              &quot;&quot;&quot;</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">              module {</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">                func.func @main(%arg0: tensor&lt;3xf32&gt;, %arg1: tensor&lt;3xf32&gt;) -&gt; tensor&lt;3xf32&gt; {</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">                  %0 = stablehlo.add %arg0, %arg1 : tensor&lt;3xf32&gt;</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">                  return %0 : tensor&lt;3xf32&gt;</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">                }</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">              }</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">              &quot;&quot;&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">              Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">to_rarray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float32[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">              Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">to_rarray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float32[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">          )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ConcreteRArray{Float32, 1}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float32[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">6.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]),)</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/349c8634d6ca2d68351adc77b7459ae312b98539/src/Ops.jl#L1303-L1326" target="_blank" rel="noreferrer">source</a></p>`,4))]),i("details",o,[i("summary",null,[s[3]||(s[3]=i("a",{id:"Reactant.Ops.randexp-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T",href:"#Reactant.Ops.randexp-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T"},[i("span",{class:"jlbinding"},"Reactant.Ops.randexp")],-1)),s[4]||(s[4]=a()),n(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[5]||(s[5]=e(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">randexp</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    ::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{T}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    seed</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TracedRArray{UInt64,1}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    shape;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    algorithm</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">String</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;DEFAULT&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    location</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">mlir_stacktrace</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;rand&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@__FILE__</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@__LINE__</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Generate a random array of type <code>T</code> with the given shape and seed from an exponential distribution with rate 1. Returns a NamedTuple with the following fields:</p><ul><li><p><code>output_state</code>: The state of the random number generator after the operation.</p></li><li><p><code>output</code>: The generated array.</p></li></ul><p><strong>Arguments</strong></p><ul><li><p><code>T</code>: The type of the generated array.</p></li><li><p><code>seed</code>: The seed for the random number generator.</p></li><li><p><code>shape</code>: The shape of the generated array.</p></li><li><p><code>algorithm</code>: The algorithm to use for generating the random numbers. Defaults to &quot;DEFAULT&quot;. Other options include &quot;PHILOX&quot; and &quot;THREE_FRY&quot;.</p></li></ul><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/349c8634d6ca2d68351adc77b7459ae312b98539/src/Ops.jl#L1161-L1183" target="_blank" rel="noreferrer">source</a></p>`,6))]),i("details",E,[i("summary",null,[s[6]||(s[6]=i("a",{id:"Reactant.Ops.randn-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T",href:"#Reactant.Ops.randn-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T"},[i("span",{class:"jlbinding"},"Reactant.Ops.randn")],-1)),s[7]||(s[7]=a()),n(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[8]||(s[8]=e(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">randn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    ::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{T}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    seed</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TracedRArray{UInt64,1}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    shape;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    algorithm</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">String</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;DEFAULT&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    location</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">mlir_stacktrace</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;rand&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@__FILE__</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@__LINE__</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Generate a random array of type <code>T</code> with the given shape and seed from a standard normal distribution of mean 0 and standard deviation 1. Returns a NamedTuple with the following fields:</p><ul><li><p><code>output_state</code>: The state of the random number generator after the operation.</p></li><li><p><code>output</code>: The generated array.</p></li></ul><p><strong>Arguments</strong></p><ul><li><p><code>T</code>: The type of the generated array.</p></li><li><p><code>seed</code>: The seed for the random number generator.</p></li><li><p><code>shape</code>: The shape of the generated array.</p></li><li><p><code>algorithm</code>: The algorithm to use for generating the random numbers. Defaults to &quot;DEFAULT&quot;. Other options include &quot;PHILOX&quot; and &quot;THREE_FRY&quot;.</p></li></ul><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/349c8634d6ca2d68351adc77b7459ae312b98539/src/Ops.jl#L1118-L1141" target="_blank" rel="noreferrer">source</a></p>`,6))]),i("details",g,[i("summary",null,[s[9]||(s[9]=i("a",{id:"Reactant.Ops.rng_bit_generator-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T<:Integer",href:"#Reactant.Ops.rng_bit_generator-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T<:Integer"},[i("span",{class:"jlbinding"},"Reactant.Ops.rng_bit_generator")],-1)),s[10]||(s[10]=a()),n(t,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[11]||(s[11]=e(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rng_bit_generator</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    ::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{T}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    seed</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TracedRArray{UInt64,1}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    shape;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    algorithm</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">String</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;DEFAULT&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    location</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">mlir_stacktrace</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;rand&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@__FILE__</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@__LINE__</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Generate a random array of type <code>T</code> with the given shape and seed from a uniform random distribution between 0 and 1. Returns a NamedTuple with the following fields:</p><ul><li><p><code>output_state</code>: The state of the random number generator after the operation.</p></li><li><p><code>output</code>: The generated array.</p></li></ul><p><strong>Arguments</strong></p><ul><li><p><code>T</code>: The type of the generated array.</p></li><li><p><code>seed</code>: The seed for the random number generator.</p></li><li><p><code>shape</code>: The shape of the generated array.</p></li><li><p><code>algorithm</code>: The algorithm to use for generating the random numbers. Defaults to &quot;DEFAULT&quot;. Other options include &quot;PHILOX&quot; and &quot;THREE_FRY&quot;.</p></li></ul><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/349c8634d6ca2d68351adc77b7459ae312b98539/src/Ops.jl#L1052-L1074" target="_blank" rel="noreferrer">source</a></p>`,6))])])}const f=l(k,[["render",y]]);export{b as __pageData,f as default};
