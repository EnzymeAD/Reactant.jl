import{_ as n,C as r,c as d,o as c,j as e,a as l,a2 as o,G as s,w as i}from"./chunks/framework.DxM46CJE.js";const v=JSON.parse('{"title":"XLA","description":"","frontmatter":{},"headers":[],"relativePath":"api/xla.md","filePath":"api/xla.md","lastUpdated":null}'),p={name:"api/xla.md"},u={class:"jldocstring custom-block"},b={class:"jldocstring custom-block"},f={class:"jldocstring custom-block"};function _(m,t,k,g,y,A){const a=r("Badge");return c(),d("div",null,[t[14]||(t[14]=e("h1",{id:"xla",tabindex:"-1"},[l("XLA "),e("a",{class:"header-anchor",href:"#xla","aria-label":'Permalink to "XLA"'},"​")],-1)),e("details",u,[e("summary",null,[t[0]||(t[0]=e("a",{id:"Reactant.XLA.AllocatorStats",href:"#Reactant.XLA.AllocatorStats"},[e("span",{class:"jlbinding"},"Reactant.XLA.AllocatorStats")],-1)),t[1]||(t[1]=l()),s(a,{type:"info",class:"jlObjectType jlType",text:"Type"})]),t[3]||(t[3]=o('<p>AllocatorStats()</p><p>Contains the following fields:</p><ul><li><p><code>num_allocs</code></p></li><li><p><code>bytes_in_use</code></p></li><li><p><code>peak_bytes_in_use</code></p></li><li><p><code>largest_alloc_size</code></p></li><li><p><code>bytes_limit</code></p></li><li><p><code>bytes_reserved</code></p></li><li><p><code>peak_bytes_reserved</code></p></li><li><p><code>bytes_reservable_limit</code></p></li><li><p><code>largest_free_block_bytes</code></p></li><li><p><code>pool_bytes</code></p></li><li><p><code>peak_pool_bytes</code></p></li></ul><p>It should be constructed using the <a href="/Reactant.jl/previews/PR839/api/xla#Reactant.XLA.allocatorstats"><code>allocatorstats</code></a> function.</p>',4)),s(a,{type:"info",class:"source-link",text:"source"},{default:i(()=>t[2]||(t[2]=[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/637dac4052d4d8d5e43a606fbf479b656474f6e1/src/xla/Stats.jl#L16-L33",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e("details",b,[e("summary",null,[t[4]||(t[4]=e("a",{id:"Reactant.XLA.allocatorstats",href:"#Reactant.XLA.allocatorstats"},[e("span",{class:"jlbinding"},"Reactant.XLA.allocatorstats")],-1)),t[5]||(t[5]=l()),s(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),t[7]||(t[7]=e("p",null,"allocatorstats([device])",-1)),t[8]||(t[8]=e("p",null,[l("Return an "),e("a",{href:"/Reactant.jl/previews/PR839/api/xla#Reactant.XLA.AllocatorStats"},[e("code",null,"AllocatorStats")]),l(" instance with information about the device specific allocator.")],-1)),t[9]||(t[9]=e("div",{class:"warning custom-block"},[e("p",{class:"custom-block-title"},"Warning"),e("p",null,"This method is currently not implemented for the CPU device.")],-1)),s(a,{type:"info",class:"source-link",text:"source"},{default:i(()=>t[6]||(t[6]=[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/637dac4052d4d8d5e43a606fbf479b656474f6e1/src/xla/Stats.jl#L48-L55",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),e("details",f,[e("summary",null,[t[10]||(t[10]=e("a",{id:"Reactant.XLA.device_ordinal",href:"#Reactant.XLA.device_ordinal"},[e("span",{class:"jlbinding"},"Reactant.XLA.device_ordinal")],-1)),t[11]||(t[11]=l()),s(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),t[13]||(t[13]=o('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">device_ordinal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(device</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Given the device, return the corresponding global device ordinal in the client.</p>',2)),s(a,{type:"info",class:"source-link",text:"source"},{default:i(()=>t[12]||(t[12]=[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/637dac4052d4d8d5e43a606fbf479b656474f6e1/src/xla/Device.jl#L15-L19",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})])])}const j=n(p,[["render",_]]);export{v as __pageData,j as default};
