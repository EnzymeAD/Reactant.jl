import{_ as l,C as p,c as h,o as r,j as a,a as i,a2 as t,G as n}from"./chunks/framework.C_VspMbV.js";const b=JSON.parse('{"title":"Reactant.Ops API","description":"","frontmatter":{},"headers":[],"relativePath":"api/ops.md","filePath":"api/ops.md","lastUpdated":null}'),d={name:"api/ops.md"},k={class:"jldocstring custom-block"},o={class:"jldocstring custom-block"},c={class:"jldocstring custom-block"},E={class:"jldocstring custom-block"},g={class:"jldocstring custom-block"},y={class:"jldocstring custom-block"};function u(T,s,F,R,_,C){const e=p("Badge");return r(),h("div",null,[s[18]||(s[18]=a("h1",{id:"Reactant.Ops-API",tabindex:"-1"},[a("code",null,"Reactant.Ops"),i(" API "),a("a",{class:"header-anchor",href:"#Reactant.Ops-API","aria-label":'Permalink to "`Reactant.Ops` API {#Reactant.Ops-API}"'},"​")],-1)),s[19]||(s[19]=a("p",null,[a("code",null,"Reactant.Ops"),i(" module provides a high-level API to construct MLIR operations without having to directly interact with the different dialects.")],-1)),s[20]||(s[20]=a("p",null,[i("Currently we haven't documented all the functions in "),a("code",null,"Reactant.Ops"),i(".")],-1)),a("details",k,[a("summary",null,[s[0]||(s[0]=a("a",{id:"Reactant.Ops.gather_getindex-Union{Tuple{N}, Tuple{T}, Tuple{Reactant.TracedRArray{T, N}, Reactant.TracedRArray{Int64, 2}}} where {T, N}",href:"#Reactant.Ops.gather_getindex-Union{Tuple{N}, Tuple{T}, Tuple{Reactant.TracedRArray{T, N}, Reactant.TracedRArray{Int64, 2}}} where {T, N}"},[a("span",{class:"jlbinding"},"Reactant.Ops.gather_getindex")],-1)),s[1]||(s[1]=i()),n(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[2]||(s[2]=t('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">gather_getindex</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(src, gather_indices)</span></span></code></pre></div><p>Uses <a href="/Reactant.jl/previews/PR164/api/stablehlo#Reactant.MLIR.Dialects.stablehlo.gather-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"><code>MLIR.Dialects.stablehlo.gather</code></a> to get the values of <code>src</code> at the indices specified by <code>gather_indices</code>. If the indices are contiguous it is recommended to directly use <a href="/Reactant.jl/previews/PR164/api/stablehlo#Reactant.MLIR.Dialects.stablehlo.dynamic_slice-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}"><code>MLIR.Dialects.stablehlo.dynamic_slice</code></a> instead.</p><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/Ops.jl#L1607-L1613" target="_blank" rel="noreferrer">source</a></p>',3))]),a("details",o,[a("summary",null,[s[3]||(s[3]=a("a",{id:"Reactant.Ops.hlo_call-Tuple{Any, Vararg{Any}}",href:"#Reactant.Ops.hlo_call-Tuple{Any, Vararg{Any}}"},[a("span",{class:"jlbinding"},"Reactant.Ops.hlo_call")],-1)),s[4]||(s[4]=i()),n(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[5]||(s[5]=t(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Ops</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">hlo_call</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(mlir_code</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">String</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Vararg{AnyTracedRArray}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">...; func_name</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">String</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;main&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> NTuple{N, AnyTracedRArray}</span></span></code></pre></div><p>Given a MLIR module given as a string, calls the function identified by the <code>func_name</code> keyword parameter (default &quot;main&quot;) with the provided arguments and return a tuple for each result of the call.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@jit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ConcreteRArray{Float32, 1}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float32[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">6.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]),)</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/Ops.jl#L1425-L1448" target="_blank" rel="noreferrer">source</a></p>`,4))]),a("details",c,[a("summary",null,[s[6]||(s[6]=a("a",{id:"Reactant.Ops.randexp-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T",href:"#Reactant.Ops.randexp-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T"},[a("span",{class:"jlbinding"},"Reactant.Ops.randexp")],-1)),s[7]||(s[7]=i()),n(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[8]||(s[8]=t(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">randexp</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    ::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{T}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    seed</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TracedRArray{UInt64,1}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    shape;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    algorithm</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">String</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;DEFAULT&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    location</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">mlir_stacktrace</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;rand&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@__FILE__</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@__LINE__</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Generate a random array of type <code>T</code> with the given shape and seed from an exponential distribution with rate 1. Returns a NamedTuple with the following fields:</p><ul><li><p><code>output_state</code>: The state of the random number generator after the operation.</p></li><li><p><code>output</code>: The generated array.</p></li></ul><p><strong>Arguments</strong></p><ul><li><p><code>T</code>: The type of the generated array.</p></li><li><p><code>seed</code>: The seed for the random number generator.</p></li><li><p><code>shape</code>: The shape of the generated array.</p></li><li><p><code>algorithm</code>: The algorithm to use for generating the random numbers. Defaults to &quot;DEFAULT&quot;. Other options include &quot;PHILOX&quot; and &quot;THREE_FRY&quot;.</p></li></ul><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/Ops.jl#L1283-L1305" target="_blank" rel="noreferrer">source</a></p>`,6))]),a("details",E,[a("summary",null,[s[9]||(s[9]=a("a",{id:"Reactant.Ops.randn-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T",href:"#Reactant.Ops.randn-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T"},[a("span",{class:"jlbinding"},"Reactant.Ops.randn")],-1)),s[10]||(s[10]=i()),n(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[11]||(s[11]=t(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">randn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    ::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{T}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    seed</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TracedRArray{UInt64,1}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    shape;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    algorithm</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">String</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;DEFAULT&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    location</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">mlir_stacktrace</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;rand&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@__FILE__</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@__LINE__</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Generate a random array of type <code>T</code> with the given shape and seed from a standard normal distribution of mean 0 and standard deviation 1. Returns a NamedTuple with the following fields:</p><ul><li><p><code>output_state</code>: The state of the random number generator after the operation.</p></li><li><p><code>output</code>: The generated array.</p></li></ul><p><strong>Arguments</strong></p><ul><li><p><code>T</code>: The type of the generated array.</p></li><li><p><code>seed</code>: The seed for the random number generator.</p></li><li><p><code>shape</code>: The shape of the generated array.</p></li><li><p><code>algorithm</code>: The algorithm to use for generating the random numbers. Defaults to &quot;DEFAULT&quot;. Other options include &quot;PHILOX&quot; and &quot;THREE_FRY&quot;.</p></li></ul><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/Ops.jl#L1240-L1263" target="_blank" rel="noreferrer">source</a></p>`,6))]),a("details",g,[a("summary",null,[s[12]||(s[12]=a("a",{id:"Reactant.Ops.rng_bit_generator-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T<:Integer",href:"#Reactant.Ops.rng_bit_generator-Union{Tuple{T}, Tuple{Type{T}, Reactant.TracedRArray{UInt64, 1}, Any}} where T<:Integer"},[a("span",{class:"jlbinding"},"Reactant.Ops.rng_bit_generator")],-1)),s[13]||(s[13]=i()),n(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[14]||(s[14]=t(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rng_bit_generator</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    ::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Type{T}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    seed</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TracedRArray{UInt64,1}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    shape;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    algorithm</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">String</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;DEFAULT&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    location</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">mlir_stacktrace</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;rand&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@__FILE__</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@__LINE__</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Generate a random array of type <code>T</code> with the given shape and seed from a uniform random distribution between 0 and 1. Returns a NamedTuple with the following fields:</p><ul><li><p><code>output_state</code>: The state of the random number generator after the operation.</p></li><li><p><code>output</code>: The generated array.</p></li></ul><p><strong>Arguments</strong></p><ul><li><p><code>T</code>: The type of the generated array.</p></li><li><p><code>seed</code>: The seed for the random number generator.</p></li><li><p><code>shape</code>: The shape of the generated array.</p></li><li><p><code>algorithm</code>: The algorithm to use for generating the random numbers. Defaults to &quot;DEFAULT&quot;. Other options include &quot;PHILOX&quot; and &quot;THREE_FRY&quot;.</p></li></ul><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/Ops.jl#L1174-L1196" target="_blank" rel="noreferrer">source</a></p>`,6))]),a("details",y,[a("summary",null,[s[15]||(s[15]=a("a",{id:"Reactant.Ops.scatter_setindex-Union{Tuple{T2}, Tuple{N}, Tuple{T}, Tuple{Reactant.TracedRArray{T, N}, Reactant.TracedRArray{Int64, 2}, Reactant.TracedRArray{T2, 1}}} where {T, N, T2}",href:"#Reactant.Ops.scatter_setindex-Union{Tuple{T2}, Tuple{N}, Tuple{T}, Tuple{Reactant.TracedRArray{T, N}, Reactant.TracedRArray{Int64, 2}, Reactant.TracedRArray{T2, 1}}} where {T, N, T2}"},[a("span",{class:"jlbinding"},"Reactant.Ops.scatter_setindex")],-1)),s[16]||(s[16]=i()),n(e,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[17]||(s[17]=t('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">scatter_setindex</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dest, scatter_indices, updates)</span></span></code></pre></div><p>Uses <a href="/Reactant.jl/previews/PR164/api/stablehlo#Reactant.MLIR.Dialects.stablehlo.scatter-Tuple{Vector{Reactant.MLIR.IR.Value}, Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}"><code>MLIR.Dialects.stablehlo.scatter</code></a> to set the values of <code>dest</code> at the indices specified by <code>scatter_indices</code> to the values in <code>updates</code>. If the indices are contiguous it is recommended to directly use <a href="/Reactant.jl/previews/PR164/api/stablehlo#Reactant.MLIR.Dialects.stablehlo.dynamic_update_slice-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}"><code>MLIR.Dialects.stablehlo.dynamic_update_slice</code></a> instead.</p><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/Ops.jl#L1543-L1550" target="_blank" rel="noreferrer">source</a></p>',3))])])}const f=l(d,[["render",u]]);export{b as __pageData,f as default};
