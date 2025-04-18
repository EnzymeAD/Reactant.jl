import{_ as o,C as c,c as r,o as d,j as s,a as l,a2 as n,G as e,w as i}from"./chunks/framework.ggWhhDUr.js";const E=JSON.parse('{"title":"XLA","description":"","frontmatter":{},"headers":[],"relativePath":"api/xla.md","filePath":"api/xla.md","lastUpdated":null}'),p={name:"api/xla.md"},u={class:"jldocstring custom-block"},k={class:"jldocstring custom-block"},h={class:"jldocstring custom-block"},b={class:"jldocstring custom-block"};function f(y,t,_,g,m,A){const a=c("Badge");return d(),r("div",null,[t[18]||(t[18]=s("h1",{id:"xla",tabindex:"-1"},[l("XLA "),s("a",{class:"header-anchor",href:"#xla","aria-label":'Permalink to "XLA"'},"​")],-1)),s("details",u,[s("summary",null,[t[0]||(t[0]=s("a",{id:"Reactant.XLA.AllocatorStats",href:"#Reactant.XLA.AllocatorStats"},[s("span",{class:"jlbinding"},"Reactant.XLA.AllocatorStats")],-1)),t[1]||(t[1]=l()),e(a,{type:"info",class:"jlObjectType jlType",text:"Type"})]),t[3]||(t[3]=n('<p>AllocatorStats()</p><p>Contains the following fields:</p><ul><li><p><code>num_allocs</code></p></li><li><p><code>bytes_in_use</code></p></li><li><p><code>peak_bytes_in_use</code></p></li><li><p><code>largest_alloc_size</code></p></li><li><p><code>bytes_limit</code></p></li><li><p><code>bytes_reserved</code></p></li><li><p><code>peak_bytes_reserved</code></p></li><li><p><code>bytes_reservable_limit</code></p></li><li><p><code>largest_free_block_bytes</code></p></li><li><p><code>pool_bytes</code></p></li><li><p><code>peak_pool_bytes</code></p></li></ul><p>It should be constructed using the <a href="/Reactant.jl/v0.2.57/api/xla#Reactant.XLA.allocatorstats"><code>allocatorstats</code></a> function.</p>',4)),e(a,{type:"info",class:"source-link",text:"source"},{default:i(()=>t[2]||(t[2]=[s("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/f3e0d0f12705c284c2cafd6c4acfc87b112d9751/src/xla/Stats.jl#L16-L33",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",k,[s("summary",null,[t[4]||(t[4]=s("a",{id:"Reactant.XLA.allocatorstats",href:"#Reactant.XLA.allocatorstats"},[s("span",{class:"jlbinding"},"Reactant.XLA.allocatorstats")],-1)),t[5]||(t[5]=l()),e(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),t[7]||(t[7]=s("p",null,"allocatorstats([device])",-1)),t[8]||(t[8]=s("p",null,[l("Return an "),s("a",{href:"/Reactant.jl/v0.2.57/api/xla#Reactant.XLA.AllocatorStats"},[s("code",null,"AllocatorStats")]),l(" instance with information about the device specific allocator.")],-1)),t[9]||(t[9]=s("div",{class:"warning custom-block"},[s("p",{class:"custom-block-title"},"Warning"),s("p",null,"This method is currently not implemented for the CPU device.")],-1)),e(a,{type:"info",class:"source-link",text:"source"},{default:i(()=>t[6]||(t[6]=[s("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/f3e0d0f12705c284c2cafd6c4acfc87b112d9751/src/xla/Stats.jl#L69-L77",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",h,[s("summary",null,[t[10]||(t[10]=s("a",{id:"Reactant.XLA.cost_analysis",href:"#Reactant.XLA.cost_analysis"},[s("span",{class:"jlbinding"},"Reactant.XLA.cost_analysis")],-1)),t[11]||(t[11]=l()),e(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),t[13]||(t[13]=n(`<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cost_analysis</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractLoadedExecutable</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cost_analysis</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Reactant.Thunk</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Returns a HloCostAnalysisProperties object with the cost analysis of the loaded executable.</p>`,2)),e(a,{type:"info",class:"source-link",text:"source"},{default:i(()=>t[12]||(t[12]=[s("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/f3e0d0f12705c284c2cafd6c4acfc87b112d9751/src/xla/Stats.jl#L137-L142",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),s("details",b,[s("summary",null,[t[14]||(t[14]=s("a",{id:"Reactant.XLA.device_ordinal",href:"#Reactant.XLA.device_ordinal"},[s("span",{class:"jlbinding"},"Reactant.XLA.device_ordinal")],-1)),t[15]||(t[15]=l()),e(a,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),t[17]||(t[17]=n('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">device_ordinal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(device</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Given the device, return the corresponding global device ordinal in the client.</p>',2)),e(a,{type:"info",class:"source-link",text:"source"},{default:i(()=>t[16]||(t[16]=[s("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/f3e0d0f12705c284c2cafd6c4acfc87b112d9751/src/xla/Device.jl#L15-L19",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})])])}const v=o(p,[["render",f]]);export{E as __pageData,v as default};
