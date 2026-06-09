import{_ as a,c as e,o as n,al as i}from"./chunks/framework.Dr0tXxYB.js";const d=JSON.parse('{"title":"Automatic Sharding-based Distributed Parallelism","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/sharding.md","filePath":"tutorials/sharding.md","lastUpdated":null}'),t={name:"tutorials/sharding.md"};function l(p,s,r,h,o,k){return n(),e("div",null,[...s[0]||(s[0]=[i(`<h1 id="sharding" tabindex="-1">Automatic Sharding-based Distributed Parallelism <a class="header-anchor" href="#sharding" aria-label="Permalink to &quot;Automatic Sharding-based Distributed Parallelism {#sharding}&quot;">​</a></h1><div class="tip custom-block"><p class="custom-block-title">Use XLA IFRT Runtime</p><p>While PJRT does support some minimal sharding capabilities on CUDA GPUs, sharding support in Reactant is primarily provided via IFRT. Before loading Reactant, set the &quot;xla_runtime&quot; preference to be &quot;IFRT&quot;. This can be done with:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Preferences, UUIDs</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Preferences</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">set_preferences!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    UUID</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;3c362404-f566-11ee-1572-e11a4b42c853&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">    &quot;xla_runtime&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =&gt;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;IFRT&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div></div><h2 id="Basics" tabindex="-1">Basics <a class="header-anchor" href="#Basics" aria-label="Permalink to &quot;Basics {#Basics}&quot;">​</a></h2><p>Sharding is one mechanism supported within Reactant that tries to make it easy to program for multiple devices (including <a href="/Reactant.jl/previews/PR2006/tutorials/multihost#distributed">multiple nodes</a>).</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">devices</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">8-element Vector{Reactant.XLA.PJRT.Device}:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x0000000019ae6400, &quot;CPU:0 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x000000001b5e1780, &quot;CPU:1 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x000000001ad018f0, &quot;CPU:2 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x00000000193c1d80, &quot;CPU:3 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x0000000019361ea0, &quot;CPU:4 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x0000000019197f80, &quot;CPU:5 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x0000000019345a50, &quot;CPU:6 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x000000001926e7d0, &quot;CPU:7 cpu&quot;)</span></span></code></pre></div><p>Sharding provides Reactant users a <a href="https://en.wikipedia.org/wiki/Partitioned_global_address_space" target="_blank" rel="noreferrer">PGAS (parallel-global address space)</a> programming model. Let&#39;s understand what this means through example.</p><p>Suppose we have a function that takes a large input array and computes sin for all elements of the array.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> big_sin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(data)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data .</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(data)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">N </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1600</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">to_rarray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float32, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">N), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">40</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">40</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">compiled_big_sin </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> big_sin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">compiled_big_sin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.broadcasted)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1af44890)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.broadcasted),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.broadcastable)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1af44ef0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.broadcastable),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.map)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1af451f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.map),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.broadcastable),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(tuple)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1af45400)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=tuple),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core._apply_iterate)(), Array{Any, 1}(dims=(5,), mem=Memory{Any}(5, 0x7fcefa37a3e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core._apply_iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.combine_styles),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Float32, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=())]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.combine_styles)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1af454f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.combine_styles),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(tuple)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcefa37a460)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=tuple),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core._apply_iterate)(), Array{Any, 1}(dims=(5,), mem=Memory{Any}(5, 0x7fcefa37a4e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core._apply_iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.broadcasted),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, typeof(Base.sin), Reactant.TracedRArray{Float32, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=())]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.broadcasted)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcefa37a520)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.broadcasted),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  typeof(Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}, Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcefa37abe0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Float32, 2}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1afabec0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}())]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1afabf20)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=nothing)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1afabf80)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:Typeof)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getglobal)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b0c8380)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getglobal),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:Typeof)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.Typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1b0c8980)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.Typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1b0c8bc0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Float32, 2}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(6,), mem=Memory{Any}(6, 0x7fcef483fa40)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Nothing),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof(Base.sin)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Reactant.TracedRArray{Float32, 2}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b0c8cb0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=1)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b0c8d40)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b0c8da0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b0c8e30)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof(Base.sin))]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b0c8e90)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=3)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b0c8f20)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Float32, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Reactant.TracedRArray{Float32, 2}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b0c8f80)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=4)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b0c9010)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=nothing),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Nothing)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.materialize!)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b1d1190)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.materialize!),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.combine_styles)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b1d15e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.combine_styles),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.combine_styles)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1b1d1a00)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.combine_styles),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.combine_styles)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1b1d1cd0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.combine_styles),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b1d2b70)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:style)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b1d2f00)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Symbol]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b1d3bf0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:style)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcefa2ce020)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Union),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Nothing),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Unknown)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b1d3d10)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Union{Base.Broadcast.Unknown, Nothing})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b1d3d70)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:style)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b254320)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:style)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.result_style)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b254b00)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.result_style),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}())]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.materialize!)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcefa2cf4a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.materialize!),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b2557f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b2558b0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b255e20)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b255f10)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b256480)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.axes)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1b2565a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Axes, F, Args} where Args&lt;:Tuple where F where Axes, Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcefa223560)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Axes, F, Args} where Args&lt;:Tuple where F where Axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Float32, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1ab9c8c0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1ab9c920)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:Typeof)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.Typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1ab9cc50)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.Typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1ab9d0a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Float32, 2}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(6,), mem=Memory{Any}(6, 0x7fcef484ac80)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof(Base.sin)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Reactant.TracedRArray{Float32, 2}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1ab9d190)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=1)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf17057480)[Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeassert)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1ab9d340)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeassert),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1ab9d4c0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1ab9d520)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1ab9d5b0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof(Base.sin))]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1ab9d610)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=3)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1ab9d6a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Float32, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Reactant.TracedRArray{Float32, 2}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1ab9d820)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=4)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1ab9d8e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.instantiate)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf19e06060)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.instantiate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19e06a50)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19e06ed0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Symbol]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19d353d0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19d354c0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Nothing)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19d35520)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19d35b50)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(tuple)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf19d35d60)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=tuple),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19d35df0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19d36780)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core._apply_iterate)(), Array{Any, 1}(dims=(5,), mem=Memory{Any}(5, 0x7fcef7f41120)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core._apply_iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.check_broadcast_axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Float32, 2}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.check_broadcast_axes)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(8, 0x7fcf195c86e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.check_broadcast_axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  #&lt;null&gt;,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  #&lt;null&gt;,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  #&lt;null&gt;,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  #&lt;null&gt;,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  #&lt;null&gt;]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19d36cf0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:style)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19d377a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:style)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19d37950)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19d37f20)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19d10710)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19b61190)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}, Array{Any, 1}(dims=(5,), mem=Memory{Any}(5, 0x7fcef7f41620)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Float32, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf19a5da30)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}())]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf19a5dac0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19a5db20)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:Typeof)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.Typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf19a5deb0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.Typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf19a5e240)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Float32, 2}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(6,), mem=Memory{Any}(6, 0x7fcef48b0d90)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof(Base.sin)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Reactant.TracedRArray{Float32, 2}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19a5e390)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=1)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19a5e420)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19a5e480)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19a5e540)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof(Base.sin))]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19a5e5a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=3)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19a5e630)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Float32, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Reactant.TracedRArray{Float32, 2}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19a5e690)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=4)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf19a5e720)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Reactant.TracedRArrayOverrides._copyto!)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1ab29f40)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides._copyto!),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.axes)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1ab2a9f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.axes)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1ab2ac90)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1ab2b1a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1ab2b740)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast._axes)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1ab2b830)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast._axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.:(var&quot;==&quot;))(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aaed340)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.:(var&quot;==&quot;)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.axes)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1aaed5e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.axes)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1aaed970)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.throwdm)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aaedc40)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.throwdm),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.isempty)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1aaede20)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.isempty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.preprocess)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aaee090)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.preprocess),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aaee450)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:style)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aaeef30)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:style)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aaef110)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aaef590)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aaef680)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aaefe30)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.preprocess_args)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aac4080)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.preprocess_args),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Float32, 2}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aac42c0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aac4920)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Float32, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}, Array{Any, 1}(dims=(5,), mem=Memory{Any}(5, 0x7fcef8b94060)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcfc57023a0)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcfc4c34020)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1aac6750)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}())]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1aac67b0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aac6810)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:Typeof)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.Typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1aac6a50)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.Typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1aac6c60)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(6,), mem=Memory{Any}(6, 0x7fcef48b3680)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof(Base.sin)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aac6db0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=1)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aac75f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aac7680)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aac7740)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof(Base.sin))]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aac77a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=3)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aac7c80)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aac7ce0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=4)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aac7e00)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1aa0f290)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}, fields=Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef78d8360)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf171f7720)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aa58cb0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aa0f320)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{bc} where bc),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aa0f4a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}, fields=Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef78d8360)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf171f7720)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aa58cb0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a9e4d10)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Symbol]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a9e5910)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}, fields=Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef78d8360)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf171f7720)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aa58cb0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, Base.Generator{I, F} where F where I, Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a9e5d60)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Generator{I, F} where F where I),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf1726c260)[Core.PartialStruct(typ=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}, fields=Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef78d8360)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf171f7720)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aa58cb0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf171f7720)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aa58cb0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))]))]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef78da7a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Generator{I, F} where F where I),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Reactant.unwrapped_eltype)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1a9b4710)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.unwrapped_eltype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.ndims)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1a9b4ef0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.ndims),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef78daf60)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArray{T, N} where N where T),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Float32),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a9b57f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}, fields=Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef78d8360)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf171f7720)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aa58cb0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a9b6300)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}, fields=Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef78d8360)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf171f7720)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aa58cb0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(tuple)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1a9b63f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=tuple),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core._apply_iterate)(), Array{Any, 1}(dims=(5,), mem=Memory{Any}(5, 0x7fcef78db120)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core._apply_iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedUtils.elem_apply),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(typeof(Base.sin)(),)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Base.Generator{Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}}}, fields=Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1a9b4590)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf1726c260)[Core.PartialStruct(typ=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}, fields=Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef78d8360)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf171f7720)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aa58cb0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf171f7720)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aa58cb0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))]))]))]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.iterate)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1a9b64b0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Base.Generator{Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}}}, fields=Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1a9b4590)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf1726c260)[Core.PartialStruct(typ=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}, fields=Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef78d8360)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf171f7720)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aa58cb0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf171f7720)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1aa58cb0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Float32, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))]))]))]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a9b72f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Generator{Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:iter)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a9881d0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Generator{Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Symbol]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a989370)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Generator{Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:iter)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(tuple)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1a989490)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=tuple),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core._apply_iterate)(), Array{Any, 1}(dims=(5,), mem=Memory{Any}(5, 0x7fcefabccce0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core._apply_iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.iterate)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1a989640)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(var&quot;===&quot;)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a989940)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=var&quot;===&quot;),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, Int64}, fields=Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf18339970)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=nothing)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcefabccd60)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Any),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Any)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeassert)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a989a00)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeassert),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, Int64}, fields=Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf18339970)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Any, Any})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a989a90)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Generator{Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a989f70)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Generator{Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getindex)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a98a150)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getindex),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, Int64}, fields=Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf18339970)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=1)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcefabcd120)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, Int64}, fields=Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf18339970)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=1),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Bool]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.materialize)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1a98b3e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.materialize),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a98b920)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:bc)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.size)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1a98b9b0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.size),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.axes)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1a960020)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a9603e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a9609e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast._axes)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a960ad0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast._axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.map)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a962240)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.map),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.length),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Reactant.broadcast_to_size)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a963590)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.broadcast_to_size),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Int64, Int64}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getindex)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1a940a40)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getindex),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, Int64}, fields=Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf18339970)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef7d396e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Float32, 2}, Tuple{Bool, Bool}, Tuple{Int64, }</span></span></code></pre></div><p>This successfully allocates the array <code>x</code> on one device, and executes it on the same device. However, suppose we want to execute this computation on multiple devices. Perhaps this is because the size of our inputs (<code>N</code>) is too large to fit on a single device. Or alternatively the function we execute is computationally expensive and we want to leverage the computing power of multiple devices.</p><p>Unlike more explicit communication libraries like MPI, the sharding model used by Reactant aims to let you execute a program on multiple devices without significant modifications to the single-device program. In particular, you do not need to write explicit communication calls (e.g. <code>MPI.Send</code> or <code>MPI.Recv</code>). Instead you write your program as if it executes on a very large single-node and Reactant will automatically determine how to subdivide the data, computation, and required communication.</p><p>When using sharding, the one thing you need to change about your code is how arrays are allocated. In particular, you need to specify how the array is partitioned amongst available devices. For example, suppose you are on a machine with 4 GPUs. In the example above, we computed <code>sin</code> for all elements of a 40x40 grid. One partitioning we could select is to have it partitioned along the first axis, such that each GPU has a slice of 10x40 elements. We could accomplish this as follows. No change is required to the original function. However, the compiled function is specific to the sharding so we need to compile a new version for our sharded array.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">N </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1600</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x_sharded_first </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">to_rarray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">N), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">40</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">40</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    sharding</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Sharding</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedSharding</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        Sharding</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Mesh</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">devices</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">compiled_big_sin_sharded_first </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> big_sin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_sharded_first)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">compiled_big_sin_sharded_first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_sharded_first)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.broadcasted)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f670470)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.broadcasted),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.broadcastable)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf6f670950)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.broadcastable),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.map)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f671670)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.map),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.broadcastable),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(tuple)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf6f6718b0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=tuple),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core._apply_iterate)(), Array{Any, 1}(dims=(5,), mem=Memory{Any}(5, 0x7fcef576a5e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core._apply_iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.combine_styles),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Int64, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=())]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.combine_styles)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf6f671940)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.combine_styles),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf6f671cd0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, Base.Broadcast.BroadcastStyle, Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf6f671d30)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.BroadcastStyle),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArray{Int64, 2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f6721e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{N} where N),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf1ba81740)[Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.result_style)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf6f6729f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.result_style),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}())]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(tuple)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef5764760)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=tuple),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core._apply_iterate)(), Array{Any, 1}(dims=(5,), mem=Memory{Any}(5, 0x7fcef57647e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core._apply_iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.broadcasted),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, typeof(Base.sin), Reactant.TracedRArray{Int64, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=())]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.broadcasted)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef5764820)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.broadcasted),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  typeof(Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}, Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef5764ea0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Int64, 2}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf6f0dd940)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}())]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf6f0dd9a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=nothing)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f0dda00)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:Typeof)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getglobal)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f0deff0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getglobal),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:Typeof)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.Typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf6f0df680)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.Typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf6f0df890)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Int64, 2}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(6,), mem=Memory{Any}(6, 0x7fcefaa2e5f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Nothing),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof(Base.sin)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Reactant.TracedRArray{Int64, 2}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f0df980)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=1)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f0dfa10)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f0dfa70)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f0dfb00)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof(Base.sin))]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f0dfb90)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=3)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f0dfc20)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Int64, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Reactant.TracedRArray{Int64, 2}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f0dfc80)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=4)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f0dfd10)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=nothing),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Nothing)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.materialize!)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f2b73e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.materialize!),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.combine_styles)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f2b79e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.combine_styles),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.combine_styles)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf6f2b7e30)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.combine_styles),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.combine_styles)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf6f414020)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.combine_styles),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f4144a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:style)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f414a40)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Symbol]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f415850)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:style)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef5753c20)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Union),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Nothing),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Unknown)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f415ac0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Union{Base.Broadcast.Unknown, Nothing})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f415b20)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:style)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f4160f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:style)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.result_style)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f416a50)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.result_style),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}())]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.materialize!)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef573d020)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.materialize!),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f417830)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f417980)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f518620)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f518740)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f518d10)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Nothing, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.axes)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf6f518e00)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.size)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf6f519430)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.size),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.map)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf6f519a90)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.map),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.unchecked_oneto),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Int64, Int64}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Axes, F, Args} where Args&lt;:Tuple where F where Axes, Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef573f1e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Axes, F, Args} where Args&lt;:Tuple where F where Axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Int64, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1b99b5c0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b99b620)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:Typeof)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.Typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1b99b800)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.Typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1b99ba70)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Int64, 2}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(6,), mem=Memory{Any}(6, 0x7fcef9961920)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof(Base.sin)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Reactant.TracedRArray{Int64, 2}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b99bb90)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=1)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf6de2bc60)[Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeassert)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b99bda0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeassert),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b99be30)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b99be90)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b99bf20)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof(Base.sin))]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b99bf80)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=3)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b97c020)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Int64, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Reactant.TracedRArray{Int64, 2}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b97c0b0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=4)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b97c140)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.instantiate)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1b97c8f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.instantiate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b97cf20)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b97d2b0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Symbol]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b97dee0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b97dfd0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Nothing)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b97e030)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b97e480)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(tuple)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1b97e5a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=tuple),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b97e600)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b97eab0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core._apply_iterate)(), Array{Any, 1}(dims=(5,), mem=Memory{Any}(5, 0x7fcef6412d20)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core._apply_iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.check_broadcast_axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Int64, 2}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.check_broadcast_axes)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(8, 0x7fcf19c600e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.check_broadcast_axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  #&lt;null&gt;,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  #&lt;null&gt;,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  #&lt;null&gt;,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  #&lt;null&gt;,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  #&lt;null&gt;]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.axes)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1b97ef90)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.check_broadcast_shape)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b97f140)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.check_broadcast_shape),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b96d9a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:style)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b96dfa0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:style)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b96e090)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b96e6f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b96e7e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1b96ec60)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}, Array{Any, 1}(dims=(5,), mem=Memory{Any}(5, 0x7fcef6418be0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Int64, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf18018260)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}())]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf180182f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf18018380)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:Typeof)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.Typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf18018560)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.Typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf18018710)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Int64, 2}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(6,), mem=Memory{Any}(6, 0x7fcef99634f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof(Base.sin)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Reactant.TracedRArray{Int64, 2}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf180187a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=1)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf18018830)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf18018890)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf18018920)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof(Base.sin))]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf18018980)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=3)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf18018a10)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Int64, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Reactant.TracedRArray{Int64, 2}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf18018b60)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=4)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf18018bf0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Reactant.TracedRArrayOverrides._copyto!)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1803c410)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides._copyto!),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.axes)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1803d700)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.axes)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1803d8b0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1803dfa0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1803efc0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast._axes)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1803f0b0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast._axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.:(var&quot;==&quot;))(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1805c500)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.:(var&quot;==&quot;)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.axes)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1805c7a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.axes)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1805c950)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.throwdm)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1805cb00)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.throwdm),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.isempty)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1805cce0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.isempty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.length)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1805d0d0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.length),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.size)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1805d430)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.size),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.prod)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1805d640)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.prod),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Int64, Int64}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.:(var&quot;==&quot;))(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1805e9f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.:(var&quot;==&quot;)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Int64,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=0)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.preprocess)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1805ff50)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.preprocess),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1807c410)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:style)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1807c890)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:style)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1807c980)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1807cf50)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1807d040)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1807d580)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.preprocess_args)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1807d670)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.preprocess_args),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Int64, 2}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getindex)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1807da90)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getindex),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Int64, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=1)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef64d6de0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Int64, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Int64,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Bool]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef64d76a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Int64, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=1),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Bool]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.preprocess)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1807ebd0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.preprocess),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.broadcast_unalias)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1807ef60)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.broadcast_unalias),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(var&quot;===&quot;)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1807f2f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=var&quot;===&quot;),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.unalias)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1807f350)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.unalias),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.mightalias)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1807f740)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.mightalias),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.isbits)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1807fce0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.isbits),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1809db80)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.isbitstype)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1809dcd0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.isbitstype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArray{Int64, 2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.:(!))(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1809e540)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.:(!)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=false)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.isbits)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1809e870)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.isbits),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.:(!))(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1809ea80)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.:(!)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=false)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.isempty)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1809ed50)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.isempty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.:(!))(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1809ef90)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.:(!)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Bool]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.isempty)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1809f2c0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.isempty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.:(!))(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1809f530)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.:(!)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Bool]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.dataids)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1809f7a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.dataids),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.objectid)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1809fd40)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.objectid),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, UInt64, Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf180b08f0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=UInt64),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  UInt64]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(tuple)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf180b0aa0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=tuple),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  UInt64]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.dataids)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf180b1850)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.dataids),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base._isdisjoint)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf180b1b20)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base._isdisjoint),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{UInt64},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{UInt64}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.:(!))(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf180b1dc0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.:(!)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Bool]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.unaliascopy)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf180d5910)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.unaliascopy),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf180d5df0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.copy)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf180d5e50)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.copy),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef64e6420)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArray{T, N} where N where T),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Int64),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf180d62d0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:mlir_data)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf180d6690)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:mlir_data)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.size)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf180d6f60)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.size),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, Reactant.TracedRArray{Int64, 2}, Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef64e72a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArray{Int64, 2}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Union{Nothing, Reactant.MLIR.IR.Value},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Int64, Int64}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, Tuple, Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf180d7ad0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Int64, Int64}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.isnothing)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf180d7fe0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.isnothing),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=nothing)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.:(!))(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf180fc230)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.:(!)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=true)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef64f11a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArray{T, N} where N where T),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Int64),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf180fc4d0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArray{Int64, 2}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=1)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf180fc590)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Type{Tuple}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf180fc620)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArray{Int64, 2}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf180fc710)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=nothing),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Type{Union{Nothing, Reactant.MLIR.IR.Value}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf180fc770)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArray{Int64, 2}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=3)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf180fc830)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Int64, Int64},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Int64, Int64})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base._unaliascopy)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf180fec30)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base._unaliascopy),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf180ff620)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArray{Int64, 2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.extrude)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1811e8a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.extrude),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(tuple)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1811ed50)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=tuple),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.newindexer)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1811ee10)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.newindexer),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.axes)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1811f4a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.axes),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.Broadcast.shapeindexer)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1811f7a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.shapeindexer),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core._apply_iterate)(), Array{Any, 1}(dims=(5,), mem=Memory{Any}(5, 0x7fcef650be60)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core._apply_iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Extruded{T, K, D} where D where K where T),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Reactant.TracedRArray{Int64, 2}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1811fec0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, Base.Broadcast.Extruded{T, K, D} where D where K where T, Array{Any, 1}(dims=(4,), mem=Memory{Any}(8, 0x7fcf1a6c1040)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Extruded{T, K, D} where D where K where T),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  #&lt;null&gt;,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  #&lt;null&gt;,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  #&lt;null&gt;,</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  #&lt;null&gt;]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(5,), mem=Memory{Any}(5, 0x7fcef6514b20)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Extruded{T, K, D} where D where K where T),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArray{Int64, 2}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Bool, Bool}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Int64, Int64})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(tuple)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1813db80)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=tuple),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1812be30)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1813f7a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1813fbf0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Reactant.TracedRArray{Int64, 2}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:axes)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}, Array{Any, 1}(dims=(5,), mem=Memory{Any}(5, 0x7fcef6517da0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf6d311a00)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1813dbe0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1814d2e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}())]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1814d340)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1814d3a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:Typeof)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.Typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1814d580)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.Typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf1814d730)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(6,), mem=Memory{Any}(6, 0x7fcef699fb30)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Style, Axes, F, Args} where Args&lt;:Tuple where F where Axes where Style&lt;:Union{Nothing, Base.Broadcast.BroadcastStyle}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof(Base.sin)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1814d820)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=1)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1814d8b0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1814d910)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1814d9a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof(Base.sin))]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1814da00)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=3)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1814dbb0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(fieldtype)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1814dc40)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=fieldtype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=4)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(isa)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1814dd00)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=isa),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeof)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf18160ce0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeof),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}, fields=Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef65293e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf6cd5e2c0)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1814e750)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf18160d40)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{bc} where bc),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf18160dd0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}, fields=Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef65293e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf6cd5e2c0)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1814e750)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf18161130)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Symbol]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf18161d90)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}, fields=Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef65293e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf6cd5e2c0)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1814e750)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:args)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, Base.Generator{I, F} where F where I, Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf18161e80)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Generator{I, F} where F where I),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf6cc7a840)[Core.PartialStruct(typ=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}, fields=Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef65293e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf6cd5e2c0)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1814e750)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf6cd5e2c0)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1814e750)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))]))]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef652b3a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.Generator{I, F} where F where I),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Reactant.unwrapped_eltype)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf181805c0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.unwrapped_eltype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Reactant.unwrapped_eltype)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf18180b00)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.unwrapped_eltype),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRNumber{Int64})]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.ndims)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf18181760)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.ndims),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2}]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(typeassert)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf18181be0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=typeassert),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Int64)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core.apply_type)(), Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef65315e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core.apply_type),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArray{T, N} where N where T),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Int64),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=2)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Base.getproperty)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf181824e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.getproperty),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}, fields=Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef65293e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf6cd5e2c0)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1814e750)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(getfield)(), Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf18182960)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=getfield),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}, fields=Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef65293e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf6cd5e2c0)[Core.PartialStruct(typ=Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}, fields=Array{Any, 1}(dims=(3,), mem=Memory{Any}(3, 0x7fcf1814e750)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Reactant.TracedRArray{Int64, 2},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Bool, Bool},</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(1, 1))]))])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}])),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=:f)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(tuple)(), Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf18182a80)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=tuple),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.sin)]))</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;SET_REACTANT_ABI&quot;, typeof(Core._apply_iterate)(), Array{Any, 1}(dims=(5,), mem=Memory{Any}(5, 0x7fcef6531820)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Core._apply_iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Base.iterate),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedUtils.elem_apply),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=(typeof(Base.sin)(),)),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Base.Generator{Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}, Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}}}, fields=Array{Any, 1}(dims=(2,), mem=Memory{Any}(2, 0x7fcf181804a0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.PartialStruct(typ=Reactant.TracedRArrayOverrides.var&quot;#15#16&quot;{Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}}, fields=Array{Any, 1}(dims=(1,), mem=Memory{Any}(1, 0x7fcf6cc7a840)[Core.PartialStruct(typ=Base.Broadcast.Broadcasted{Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.sin), Tuple{Base.Broadcast.Extruded{Reactant.TracedRArray{Int64, 2}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}, fields=Array{Any, 1}(dims=(4,), mem=Memory{Any}(4, 0x7fcef65293e0)[</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  Core.Const(val=Reactant.TracedRArrayOverrides.AbstractReactantArrayStyle{2}()),</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  .(=</span></span></code></pre></div><p>Alternatively, we can parition the data in a different form. In particular, we could subdivide the data on both axes. As a result each GPU would have a slice of 20x20 elements. Again no change is required to the original function, but we would change the allocation as follows:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">N </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1600</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x_sharded_both </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">to_rarray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">N), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">40</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">40</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    sharding</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Sharding</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedSharding</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        Sharding</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Mesh</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">devices</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">compiled_big_sin_sharded_both </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> big_sin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_sharded_both)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">compiled_big_sin_sharded_both</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_sharded_both)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">I0000 00:00:1766421198.278508    2746 shardy_xla_pass.cc:318] Using Shardy for XLA SPMD propagation.</span></span></code></pre></div><p>Sharding in reactant requires you to specify how the data is sharded across devices on a mesh. We start by specifying the mesh <a href="/Reactant.jl/previews/PR2006/api/sharding#Reactant.Sharding.Mesh"><code>Sharding.Mesh</code></a> which is a collection of the devices reshaped into an N-D grid. Additionally, we can specify names for each axis of the mesh, that are then referenced when specifying how the data is sharded.</p><ol><li><p><code>Sharding.Mesh(reshape(Reactant.devices()[1:4], 2, 2), (:x, :y))</code>: Creates a 2D grid of 4 devices arranged in a 2x2 grid. The first axis is named <code>:x</code> and the second axis is named <code>:y</code>.</p></li><li><p><code>Sharding.Mesh(reshape(Reactant.devices()[1:4], 4, 1), (:x, :y))</code>: Creates a 2D grid of 4 devices arranged in a 4x1 grid. The first axis is named <code>:x</code> and the second axis is named <code>:y</code>.</p></li></ol><p>Given the mesh, we will specify how the data is sharded across the devices.</p><h2 id="Gradients" tabindex="-1">Gradients <a class="header-anchor" href="#Gradients" aria-label="Permalink to &quot;Gradients {#Gradients}&quot;">​</a></h2><p>It is also possible to compute gradients of functions that are sharded. Here we show an example using the <code>Enzyme.jl</code> package.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Enzyme</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_gradient</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_sharded_both)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Enzyme</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">gradient</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Enzyme</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ReverseWithPrimal, Enzyme</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(big_sin), x_sharded_both)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@jit</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> compute_gradient</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_sharded_both)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(derivs = (nothing,), val = nothing)</span></span></code></pre></div><p>&lt;!– TODO describe how arrays are the &quot;global data arrays, even though data is itself only stored on relevant device and computation is performed only devices with the required data (effectively showing under the hood how execution occurs) –&gt;</p><p>&lt;!– TODO make a simple conway&#39;s game of life, or heat equation using sharding simulation example to show how a \`\`typical MPI&#39;&#39; simulation can be written using sharding. –&gt;</p><h2 id="Simple-1-Dimensional-Heat-Equation" tabindex="-1">Simple 1-Dimensional Heat Equation <a class="header-anchor" href="#Simple-1-Dimensional-Heat-Equation" aria-label="Permalink to &quot;Simple 1-Dimensional Heat Equation {#Simple-1-Dimensional-Heat-Equation}&quot;">​</a></h2><p>So far we chose a function which was perfectly parallelizable (e.g. each elemnt of the array only accesses its own data). Let&#39;s consider a more realistic example where an updated element requires data from its neighbors. In the distributed case, this requires communicating the data along the boundaries.</p><p>In particular, let&#39;s implement a one-dimensional <a href="https://en.wikipedia.org/wiki/Heat_equation" target="_blank" rel="noreferrer">heat equation</a> simulation. In this code you initialize the temperature of all points of the simulation and over time the code will simulate how the heat is transfered across space. In particular points of high temperature will transfer energy to points of low energy.</p><p>As an example, here is a visualization of a 2-dimensional heat equation:</p><img src="https://upload.wikimedia.org/wikipedia/commons/a/a9/Heat_eqn.gif" alt="Heat Equation Animation"><p>&lt;!– TODO we should animate the above – and even more ideally have one we generate ourselves. –&gt;</p><p>To keep things simple, let&#39;s implement a 1-dimensional heat equation here. We start off with an array for the temperature at each point, and will compute the next version of the temperatures according to the equation <code>x[i, t] = 0.x * [i, t-1] + 0.25 * x[i-1, t-1] + 0.25 * x[i+1, t-1]</code>.</p><p>Let&#39;s consider how this can be implemented with explicit MPI communication. Each node will contain a subset of the total data. For example, if we simulate with 100 points, and have 4 devices, each device will contain 25 data points. We&#39;re going to allocate some extra room at each end of the buffer to store the \`\`halo&#39;&#39;, or the data at the boundary. Each time step that we take will first copy in the data from its neighbors into the halo via an explicit MPI send and recv call. We&#39;ll then compute the updated data for our slice of the data.</p><p>With sharding, things are a bit more simple. We can write the code as if we only had one device. No explicit send or recv&#39;s are necessary as they will be added automatically by Reactant when it deduces they are needed. In fact, Reactant will attempt to optimize the placement of the communicatinos to minimize total runtime. While Reactant tries to do a good job (which could be faster than an initial implementation – especially for complex codebases), an expert may be able to find a better placement of the communication.</p><p>The only difference for the sharded code again occurs during allocation. Here we explicitly specify that we want to subdivide the initial grid of 100 amongst all devices. Analagously if we had 4 devices to work with, each device would have 25 elements in its local storage. From the user&#39;s standpoint, however, all arrays give access to the entire dataset.</p><div class="vp-code-group vp-adaptive-theme"><div class="tabs"><input type="radio" name="group-ocA_8" id="tab-LTVKsCu" checked><label data-title="MPI Based Parallelism" for="tab-LTVKsCu">MPI Based Parallelism</label><input type="radio" name="group-ocA_8" id="tab-V8XN3M_"><label data-title="Sharded Parallelism" for="tab-V8XN3M_">Sharded Parallelism</label></div><div class="blocks"><div class="language-julia vp-adaptive-theme active"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> one_dim_heat_equation_time_step_mpi!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(data)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MPI</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Comm_rank</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(MPI</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">COMM_WORLD)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    last_id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MPI</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Comm_size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(MPI</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">COMM_WORLD)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Send data right</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        MPI</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Send</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@view</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]), MPI</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">COMM_WORLD; dest</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Recv data from left</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">!=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> last_id</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        MPI</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Recv</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@view</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]), MPI</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">COMM_WORLD; dest</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">id </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # 1-D Heat equation x[i, t] = 0.x * [i, t-1] + 0.25 * x[i-1, t-1] + 0.25 * x[i+1, t-1]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] .</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.25</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.25</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># Total size of grid we want to simulate</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">N </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># Local size of grid (total size divided by number of MPI devices)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">_local </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> N </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MPI</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Comm_size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(MPI</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">COMM_WORLD)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># We add two to add a left side padding and right side padding, necessary for storing</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># boundaries from other nodes</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(_local </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> simulate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(data, time_steps)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> i </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">time_steps</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        one_dim_heat_equation_time_step_mpi!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(data)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">simulate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(data, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> one_dim_heat_equation_time_step_sharded!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(data)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # No send recv&#39;s required</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # 1-D Heat equation x[i, t] = 0.x * [i, t-1] + 0.25 * x[i-1, t-1] + 0.25 * x[i+1, t-1]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Reactant will automatically insert send and recv&#39;s</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] .</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.25</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.25</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># Total size of grid we want to simulate</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">N </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># Reactant&#39;s sharding handles distributing the data amongst devices, with each device</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># getting a corresponding fraction of the data</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">to_rarray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(N </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    sharding</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Sharding</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedSharding</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        Sharding</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Mesh</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">devices</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,)),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> simulate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(data, time_steps)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @trace</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> i </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">time_steps</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        one_dim_heat_equation_time_step_sharded!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(data)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@jit</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> simulate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(data, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div></div></div><p>MPI to send the data. between computers When using GPUs on different devices, one needs to copy the data through the network via NCCL instead of the \`cuda.</p><p>All devices from all nodes are available for use by Reactant. Given the topology of the devices, Reactant will automatically determine the right type of communication primitive to use to send data between the relevant nodes. For example, between GPUs on the same host Reactant may use the faster <code>cudaMemcpy</code> whereas for GPUs on different nodes Reactant will use NCCL.</p><p>One nice feature about how Reactant&#39;s handling of multiple devices is that you don&#39;t need to specify how the data is transfered. The fact that you doesn&#39;t need to specify how the communication is occuring enables code written with Reactant to be run on a different topology. For example, when using multiple GPUs on the same host it might be efficient to copy data using a <code>cudaMemcpy</code> to transfer between devices directly.</p><h2 id="Devices" tabindex="-1">Devices <a class="header-anchor" href="#Devices" aria-label="Permalink to &quot;Devices {#Devices}&quot;">​</a></h2><p>You can query the available devices that Reactant can access as follows using <a href="/Reactant.jl/previews/PR2006/api/api#Reactant.devices"><code>Reactant.devices</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">devices</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">8-element Vector{Reactant.XLA.PJRT.Device}:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x0000000019ae6400, &quot;CPU:0 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x000000001b5e1780, &quot;CPU:1 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x000000001ad018f0, &quot;CPU:2 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x00000000193c1d80, &quot;CPU:3 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x0000000019361ea0, &quot;CPU:4 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x0000000019197f80, &quot;CPU:5 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x0000000019345a50, &quot;CPU:6 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x000000001926e7d0, &quot;CPU:7 cpu&quot;)</span></span></code></pre></div><p>Not all devices are accessible from each process for <a href="./@ref multihost">multi-node execution</a>. To query the devices accessible from the current process, use <a href="/Reactant.jl/previews/PR2006/api/api#Reactant.addressable_devices"><code>Reactant.addressable_devices</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">addressable_devices</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">8-element Vector{Reactant.XLA.PJRT.Device}:</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x0000000019ae6400, &quot;CPU:0 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x000000001b5e1780, &quot;CPU:1 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x000000001ad018f0, &quot;CPU:2 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x00000000193c1d80, &quot;CPU:3 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x0000000019361ea0, &quot;CPU:4 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x0000000019197f80, &quot;CPU:5 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x0000000019345a50, &quot;CPU:6 cpu&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;"> Reactant.XLA.PJRT.Device(Ptr{Nothing} @0x000000001926e7d0, &quot;CPU:7 cpu&quot;)</span></span></code></pre></div><p>You can inspect the type of the device, as well as its properties.</p><p>&lt;!– TODO: Generating Distributed Data by Concatenating Local-Worker Data –&gt;</p><p>&lt;!– TODO: Handling Replicated Tensors –&gt;</p><p>&lt;!– TODO: Sharding in Neural Networks –&gt;</p><p>&lt;!– TODO: 8-way Batch Parallelism –&gt;</p><p>&lt;!– TODO: 4-way Batch &amp; 2-way Model Parallelism –&gt;</p><h2 id="Related-links" tabindex="-1">Related links <a class="header-anchor" href="#Related-links" aria-label="Permalink to &quot;Related links {#Related-links}&quot;">​</a></h2><ol><li><p><a href="https://openxla.org/shardy" target="_blank" rel="noreferrer">Shardy Documentation</a></p></li><li><p><a href="https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html" target="_blank" rel="noreferrer">Jax Documentation</a></p></li><li><p><a href="https://jax-ml.github.io/scaling-book/sharding/" target="_blank" rel="noreferrer">Jax Scaling Book</a></p></li><li><p><a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook" target="_blank" rel="noreferrer">HuggingFace Ultra Scale Playbook</a></p></li></ol>`,56)])])}const c=a(t,[["render",l]]);export{d as __pageData,c as default};
