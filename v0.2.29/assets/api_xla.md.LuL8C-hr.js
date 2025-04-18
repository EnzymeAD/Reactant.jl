import{_ as n,C as o,c as p,o as r,j as t,a,a2 as i,G as l}from"./chunks/framework.Cbi_LnTw.js";const f=JSON.parse('{"title":"XLA","description":"","frontmatter":{},"headers":[],"relativePath":"api/xla.md","filePath":"api/xla.md","lastUpdated":null}'),c={name:"api/xla.md"},d={class:"jldocstring custom-block"},h={class:"jldocstring custom-block"},k={class:"jldocstring custom-block"};function u(b,e,_,g,y,m){const s=o("Badge");return r(),p("div",null,[e[12]||(e[12]=t("h1",{id:"xla",tabindex:"-1"},[a("XLA "),t("a",{class:"header-anchor",href:"#xla","aria-label":'Permalink to "XLA"'},"​")],-1)),t("details",d,[t("summary",null,[e[0]||(e[0]=t("a",{id:"Reactant.XLA.AllocatorStats",href:"#Reactant.XLA.AllocatorStats"},[t("span",{class:"jlbinding"},"Reactant.XLA.AllocatorStats")],-1)),e[1]||(e[1]=a()),l(s,{type:"info",class:"jlObjectType jlType",text:"Type"})]),e[2]||(e[2]=i('<p>AllocatorStats()</p><p>Contains the following fields:</p><ul><li><p><code>num_allocs</code></p></li><li><p><code>bytes_in_use</code></p></li><li><p><code>peak_bytes_in_use</code></p></li><li><p><code>largest_alloc_size</code></p></li><li><p><code>bytes_limit</code></p></li><li><p><code>bytes_reserved</code></p></li><li><p><code>peak_bytes_reserved</code></p></li><li><p><code>bytes_reservable_limit</code></p></li><li><p><code>largest_free_block_bytes</code></p></li><li><p><code>pool_bytes</code></p></li><li><p><code>peak_pool_bytes</code></p></li></ul><p>It should be constructed using the <a href="/Reactant.jl/v0.2.29/api/xla#Reactant.XLA.allocatorstats"><code>allocatorstats</code></a> function.</p><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/08968eeec3ab12ef2726c784e7f984148ca10b69/src/xla/Stats.jl#L16-L33" target="_blank" rel="noreferrer">source</a></p>',5))]),t("details",h,[t("summary",null,[e[3]||(e[3]=t("a",{id:"Reactant.XLA.allocatorstats",href:"#Reactant.XLA.allocatorstats"},[t("span",{class:"jlbinding"},"Reactant.XLA.allocatorstats")],-1)),e[4]||(e[4]=a()),l(s,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[5]||(e[5]=t("p",null,"allocatorstats([device])",-1)),e[6]||(e[6]=t("p",null,[a("Return an "),t("a",{href:"/Reactant.jl/v0.2.29/api/xla#Reactant.XLA.AllocatorStats"},[t("code",null,"AllocatorStats")]),a(" instance with information about the device specific allocator.")],-1)),e[7]||(e[7]=t("div",{class:"warning custom-block"},[t("p",{class:"custom-block-title"},"Warning"),t("p",null,"This method is currently not implemented for the CPU device.")],-1)),e[8]||(e[8]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/08968eeec3ab12ef2726c784e7f984148ca10b69/src/xla/Stats.jl#L48-L55",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",k,[t("summary",null,[e[9]||(e[9]=t("a",{id:"Reactant.XLA.device_ordinal-Tuple{Reactant.XLA.Client, Reactant.XLA.Device}",href:"#Reactant.XLA.device_ordinal-Tuple{Reactant.XLA.Client, Reactant.XLA.Device}"},[t("span",{class:"jlbinding"},"Reactant.XLA.device_ordinal")],-1)),e[10]||(e[10]=a()),l(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[11]||(e[11]=i(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">device_ordinal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(client</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Client</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, device</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">device_ordinal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(client</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Client</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, local_device_id</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Given the device or local device id, return the corresponding global device ordinal in the client.</p><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/08968eeec3ab12ef2726c784e7f984148ca10b69/src/xla/Device.jl#L18-L23" target="_blank" rel="noreferrer">source</a></p>`,3))])])}const v=n(c,[["render",u]]);export{f as __pageData,v as default};
