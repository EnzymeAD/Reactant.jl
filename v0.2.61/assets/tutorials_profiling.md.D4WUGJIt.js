import{_ as i,c as a,o as t,a2 as n}from"./chunks/framework.CozIPjlV.js";const e="/Reactant.jl/v0.2.61/assets/perfetto.BkDc_Vv7.png",l="/Reactant.jl/v0.2.61/assets/tensorboard.CbqNM2YV.png",E=JSON.parse('{"title":"Profiling","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/profiling.md","filePath":"tutorials/profiling.md","lastUpdated":null}'),p={name:"tutorials/profiling.md"};function h(r,s,o,k,d,c){return t(),a("div",null,s[0]||(s[0]=[n(`<h1 id="profiling" tabindex="-1">Profiling <a class="header-anchor" href="#profiling" aria-label="Permalink to &quot;Profiling {#profiling}&quot;">​</a></h1><h2 id="Capturing-traces" tabindex="-1">Capturing traces <a class="header-anchor" href="#Capturing-traces" aria-label="Permalink to &quot;Capturing traces {#Capturing-traces}&quot;">​</a></h2><p>When running Reactant, it is possible to capture traces using the <a href="https://jax.readthedocs.io/en/latest/profiling.html" target="_blank" rel="noreferrer">XLA profiler</a>. These traces can provide information about where the XLA specific parts of program spend time during compilation or execution. Note that tracing and compilation happen on the CPU even though the final execution is aimed to run on another device such as GPU or TPU. Therefore, including tracing and compilation in a trace will create annotations on the CPU.</p><p>Let&#39;s setup a simple function which we can then profile</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">to_rarray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">randn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float32, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">W </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">to_rarray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">randn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float32, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">b </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">to_rarray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">randn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Float32, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">linear</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, W, b) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (W </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> b</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>linear (generic function with 1 method)</span></span></code></pre></div><p>The profiler can be accessed using the <a href="/Reactant.jl/v0.2.61/api/api#Reactant.Profiler.with_profiler"><code>Reactant.with_profiler</code></a> function.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">with_profiler</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;./&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    mylinear </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> linear</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, W, b)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    mylinear</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, W, b)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>10×2 ConcretePJRTArray{Float32,2}:</span></span>
<span class="line"><span>  -5.93219   -5.40905</span></span>
<span class="line"><span> -16.1853     5.58382</span></span>
<span class="line"><span> -18.4228    -5.11777</span></span>
<span class="line"><span>  -2.3029   -18.8314</span></span>
<span class="line"><span>  15.2196   -16.0763</span></span>
<span class="line"><span>   1.66865   -6.79156</span></span>
<span class="line"><span>  -7.80648   12.1993</span></span>
<span class="line"><span>   2.09649   -4.59294</span></span>
<span class="line"><span>  -9.53894   -8.72101</span></span>
<span class="line"><span>  12.7274    -1.39885</span></span></code></pre></div><p>Running this function should create a folder called <code>plugins</code> in the folder provided to <code>Reactant.with_profiler</code> which will contain the trace files. The traces can then be visualized in different ways.</p><div class="tip custom-block"><p class="custom-block-title">Note</p><p>For more insights about the current state of Reactant, it is possible to fetch device information about allocations using the <a href="/Reactant.jl/v0.2.61/api/xla#Reactant.XLA.allocatorstats"><code>Reactant.XLA.allocatorstats</code></a> function.</p></div><h2 id="Perfetto-UI" tabindex="-1">Perfetto UI <a class="header-anchor" href="#Perfetto-UI" aria-label="Permalink to &quot;Perfetto UI {#Perfetto-UI}&quot;">​</a></h2><p><img src="`+e+`" alt=""></p><p>The first and easiest way to visualize a captured trace is to use the online <a href="https://ui.perfetto.dev/" target="_blank" rel="noreferrer"><code>perfetto.dev</code></a> tool. <a href="/Reactant.jl/v0.2.61/api/api#Reactant.Profiler.with_profiler"><code>Reactant.with_profiler</code></a> has a keyword parameter called <code>create_perfetto_link</code> which will create a usable perfetto URL for the generated trace. The function will block execution until the URL has been clicked and the trace is visualized. The URL only works once.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">with_profiler</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;./&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; create_perfetto_link</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    mylinear </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> linear</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, W, b)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    mylinear</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, W, b)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="tip custom-block"><p class="custom-block-title">Note</p><p>It is recommended to use the Chrome browser to open the perfetto URL.</p></div><h2 id="tensorboard" tabindex="-1">Tensorboard <a class="header-anchor" href="#tensorboard" aria-label="Permalink to &quot;Tensorboard&quot;">​</a></h2><p><img src="`+l+`" alt=""></p><p>Another option to visualize the generated trace files is to use the <a href="https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras" target="_blank" rel="noreferrer">tensorboard profiler plugin</a>. The tensorboard viewer can offer more details than the timeline view such as visualization for compute graphs.</p><p>First install tensorboard and its profiler plugin:</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">pip</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> install</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> tensorboard</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> tensorboard-plugin-profile</span></span></code></pre></div><p>And then run the following in the folder where the <code>plugins</code> folder was generated:</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">tensorboard</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --logdir</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> ./</span></span></code></pre></div><h2 id="Adding-Custom-Annotations" tabindex="-1">Adding Custom Annotations <a class="header-anchor" href="#Adding-Custom-Annotations" aria-label="Permalink to &quot;Adding Custom Annotations {#Adding-Custom-Annotations}&quot;">​</a></h2><p>By default, the traces contain only information captured from within XLA. The <a href="/Reactant.jl/v0.2.61/api/api#Reactant.Profiler.annotate"><code>Reactant.Profiler.annotate</code></a> function can be used to annotate traces for Julia code evaluated <em>during tracing</em>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Profiler</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">annotate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;my_annotation&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Do things...</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>The added annotations will be captured in the traces and can be seen in the different viewers along with the default XLA annotations. When the profiler is not activated, then the custom annotations have no effect and can therefore always be activated.</p>`,27)]))}const u=i(p,[["render",h]]);export{E as __pageData,u as default};
