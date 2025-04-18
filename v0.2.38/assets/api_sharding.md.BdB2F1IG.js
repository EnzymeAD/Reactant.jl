import{_ as l,C as p,c as k,o as d,j as i,a as t,a2 as e,G as n,w as h}from"./chunks/framework.C8PXbVxD.js";const _=JSON.parse('{"title":"Sharding API","description":"","frontmatter":{},"headers":[],"relativePath":"api/sharding.md","filePath":"api/sharding.md","lastUpdated":null}'),r={name:"api/sharding.md"},g={class:"jldocstring custom-block"},o={class:"jldocstring custom-block"},E={class:"jldocstring custom-block"},y={class:"jldocstring custom-block"},c={class:"jldocstring custom-block"},u={class:"jldocstring custom-block"};function F(C,s,m,b,f,A){const a=p("Badge");return d(),k("div",null,[s[24]||(s[24]=i("h1",{id:"Sharding-API",tabindex:"-1"},[t("Sharding API "),i("a",{class:"header-anchor",href:"#Sharding-API","aria-label":'Permalink to "Sharding API {#Sharding-API}"'},"​")],-1)),s[25]||(s[25]=i("p",null,[i("code",null,"Reactant.Sharding"),t(" module provides a high-level API to construct MLIR operations with support for sharding.")],-1)),s[26]||(s[26]=i("p",null,[t("Currently we haven't documented all the functions in "),i("code",null,"Reactant.Sharding"),t(".")],-1)),i("details",g,[i("summary",null,[s[0]||(s[0]=i("a",{id:"Reactant.Sharding.DimsSharding",href:"#Reactant.Sharding.DimsSharding"},[i("span",{class:"jlbinding"},"Reactant.Sharding.DimsSharding")],-1)),s[1]||(s[1]=t()),n(a,{type:"info",class:"jlObjectType jlType",text:"Type"})]),s[3]||(s[3]=e(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">DimsSharding</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    mesh</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Mesh{M}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NTuple{D,Int}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    partition_spec;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    is_closed</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NTuple{D,Bool}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ntuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Returns</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), D),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    priority</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NTuple{D,Int}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ntuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(i </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, D),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Similar to <a href="/Reactant.jl/v0.2.38/api/sharding#Reactant.Sharding.NamedSharding"><code>NamedSharding</code></a> but works for a arbitrary dimensional array. Dimensions not specified in <code>dims</code> are replicated. If any dimension in <code>dims</code> is greater than the total number of dimensions in the array, the corresponding <code>partition_spec</code>, <code>is_closed</code> and <code>priority</code> are ignored. Additionally for any negative dimensions in <code>dims</code>, the true dims are calculated as <code>ndims(x) - dim + 1</code>. A dims value of <code>0</code> will throw an error.</p>`,2)),n(a,{type:"info",class:"source-link",text:"source"},{default:h(()=>s[2]||(s[2]=[i("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/5af5c249a3fff30f5bbcaf03975bc30d7a103960/src/Sharding.jl#L248-L262",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),i("details",o,[i("summary",null,[s[4]||(s[4]=i("a",{id:"Reactant.Sharding.Mesh",href:"#Reactant.Sharding.Mesh"},[i("span",{class:"jlbinding"},"Reactant.Sharding.Mesh")],-1)),s[5]||(s[5]=t()),n(a,{type:"info",class:"jlObjectType jlType",text:"Type"})]),s[7]||(s[7]=e(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Mesh</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(devices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{XLA.AbstractDevice}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, axis_names)</span></span></code></pre></div><p>Construct a <code>Mesh</code> from an array of devices and a tuple of axis names. The size of the i-th axis is given by <code>size(devices, i)</code>. All the axis names must be unique, and cannot be nothing.</p><p><strong>Examples</strong></p><p>Assuming that we have a total of 8 devices, we can construct a mesh with the following:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> devices </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">devices</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">();</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> mesh </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Mesh</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(devices, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:z</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">));</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> mesh </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Mesh</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(devices, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">));</span></span></code></pre></div>`,5)),n(a,{type:"info",class:"source-link",text:"source"},{default:h(()=>s[6]||(s[6]=[i("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/5af5c249a3fff30f5bbcaf03975bc30d7a103960/src/Sharding.jl#L5-L23",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),i("details",E,[i("summary",null,[s[8]||(s[8]=i("a",{id:"Reactant.Sharding.NamedSharding",href:"#Reactant.Sharding.NamedSharding"},[i("span",{class:"jlbinding"},"Reactant.Sharding.NamedSharding")],-1)),s[9]||(s[9]=t()),n(a,{type:"info",class:"jlObjectType jlType",text:"Type"})]),s[11]||(s[11]=e(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedSharding</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    mesh</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Mesh</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, partition_spec</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Tuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    is_closed</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NTuple{N,Bool}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ntuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Returns</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(partition_spec)),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    priority</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NTuple{N,Int}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ntuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(i </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(partition_spec)),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Sharding annotation that indicates that the array is sharded along the given <code>partition_spec</code>. For details on the sharding representation see the <a href="https://openxla.org/shardy/sharding_representation" target="_blank" rel="noreferrer">Shardy documentation</a>.</p><p><strong>Arguments</strong></p><ul><li><p><code>mesh</code>: <a href="/Reactant.jl/v0.2.38/api/sharding#Reactant.Sharding.Mesh"><code>Sharding.Mesh</code></a> that describes the mesh of the devices.</p></li><li><p><code>partition_spec</code>: Must be equal to the ndims of the array being sharded. Each element can be:</p><ol><li><p><code>nothing</code>: indicating the corresponding dimension is replicated along the axis.</p></li><li><p>A tuple of axis names indicating the axis names that the corresponding dimension is sharded along.</p></li><li><p>A single axis name indicating the axis name that the corresponding dimension is sharded along.</p></li></ol></li></ul><p><strong>Keyword Arguments</strong></p><ul><li><p><code>is_closed</code>: A tuple of booleans indicating whether the corresponding dimension is closed along the axis. Defaults to <code>true</code> for all dimensions.</p></li><li><p><code>priority</code>: A tuple of integers indicating the priority of the corresponding dimension. Defaults to <code>-1</code> for all dimensions. A negative priority means that the priority is not considered by shardy.</p></li></ul><p><strong>Examples</strong></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> devices </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">devices</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">();</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> mesh </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Mesh</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(devices, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:z</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">));</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> sharding </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> NamedSharding</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(mesh, (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 3D Array sharded along x and y on dim 1 and 2 respectively, while dim 3 is replicated</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> sharding </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> NamedSharding</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(mesh, ((</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 3D Array sharded along x and y on dim 1, 2 and 3 are replicated</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> sharding </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> NamedSharding</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(mesh, (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># fully replicated Matrix</span></span></code></pre></div><p>See also: <a href="/Reactant.jl/v0.2.38/api/sharding#Reactant.Sharding.NoSharding"><code>Sharding.NoSharding</code></a></p>`,9)),n(a,{type:"info",class:"source-link",text:"source"},{default:h(()=>s[10]||(s[10]=[i("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/5af5c249a3fff30f5bbcaf03975bc30d7a103960/src/Sharding.jl#L118-L162",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),i("details",y,[i("summary",null,[s[12]||(s[12]=i("a",{id:"Reactant.Sharding.NoSharding",href:"#Reactant.Sharding.NoSharding"},[i("span",{class:"jlbinding"},"Reactant.Sharding.NoSharding")],-1)),s[13]||(s[13]=t()),n(a,{type:"info",class:"jlObjectType jlType",text:"Type"})]),s[15]||(s[15]=e('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NoSharding</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><p>Sharding annotation that indicates that the array is not sharded.</p><p>See also: <a href="/Reactant.jl/v0.2.38/api/sharding#Reactant.Sharding.NamedSharding"><code>Sharding.NamedSharding</code></a></p>',3)),n(a,{type:"info",class:"source-link",text:"source"},{default:h(()=>s[14]||(s[14]=[i("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/5af5c249a3fff30f5bbcaf03975bc30d7a103960/src/Sharding.jl#L90-L96",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),i("details",c,[i("summary",null,[s[16]||(s[16]=i("a",{id:"Reactant.Sharding.is_sharded-Tuple{Reactant.Sharding.NoSharding}",href:"#Reactant.Sharding.is_sharded-Tuple{Reactant.Sharding.NoSharding}"},[i("span",{class:"jlbinding"},"Reactant.Sharding.is_sharded")],-1)),s[17]||(s[17]=t()),n(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[19]||(s[19]=e(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">is_sharded</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sharding)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">is_sharded</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Checks whether the given sharding refers to no sharding.</p>`,2)),n(a,{type:"info",class:"source-link",text:"source"},{default:h(()=>s[18]||(s[18]=[i("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/5af5c249a3fff30f5bbcaf03975bc30d7a103960/src/Sharding.jl#L560-L565",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),i("details",u,[i("summary",null,[s[20]||(s[20]=i("a",{id:"Reactant.Sharding.unwrap_shardinfo-Tuple{Reactant.Sharding.AbstractSharding}",href:"#Reactant.Sharding.unwrap_shardinfo-Tuple{Reactant.Sharding.AbstractSharding}"},[i("span",{class:"jlbinding"},"Reactant.Sharding.unwrap_shardinfo")],-1)),s[21]||(s[21]=t()),n(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),s[23]||(s[23]=e('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">unwrap_shardinfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)</span></span></code></pre></div><p>Unwraps a sharding info object, returning the sharding object itself.</p>',2)),n(a,{type:"info",class:"source-link",text:"source"},{default:h(()=>s[22]||(s[22]=[i("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/5af5c249a3fff30f5bbcaf03975bc30d7a103960/src/Sharding.jl#L582-L586",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})])])}const j=l(r,[["render",F]]);export{_ as __pageData,j as default};
