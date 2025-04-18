import{_ as o,C as i,c as r,o as d,j as t,a,G as l,w as n,a2 as c}from"./chunks/framework.CnUgCHXS.js";const R=JSON.parse('{"title":"TPU Dialect","description":"","frontmatter":{},"headers":[],"relativePath":"api/dialects/tpu.md","filePath":"api/dialects/tpu.md","lastUpdated":null}'),p={name:"api/dialects/tpu.md"},u={class:"jldocstring custom-block"},m={class:"jldocstring custom-block"};function b(f,e,h,g,_,k){const s=i("Badge");return d(),r("div",null,[e[9]||(e[9]=t("h1",{id:"TPU-Dialect",tabindex:"-1"},[a("TPU Dialect "),t("a",{class:"header-anchor",href:"#TPU-Dialect","aria-label":'Permalink to "TPU Dialect {#TPU-Dialect}"'},"​")],-1)),e[10]||(e[10]=t("p",null,[a("Refer to the "),t("a",{href:"https://github.com/jax-ml/jax/blob/main/jaxlib/mosaic/dialect/tpu/tpu.td",target:"_blank",rel:"noreferrer"},"official documentation"),a(" for more details.")],-1)),t("details",u,[t("summary",null,[e[0]||(e[0]=t("a",{id:"Reactant.MLIR.Dialects.tpu.broadcast_in_sublanes-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tpu.broadcast_in_sublanes-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tpu.broadcast_in_sublanes")],-1)),e[1]||(e[1]=a()),l(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[3]||(e[3]=t("p",null,[t("code",null,"broadcast_in_sublanes")],-1)),e[4]||(e[4]=t("p",null,[a("For each sublane "),t("code",null,"i"),a(", broadcasts the value in lane "),t("code",null,"lane + i"),a(" along the entire sublane. If "),t("code",null,"lane + i"),a(" is not in [0, lane_count), then the value in sublane "),t("code",null,"i"),a(" is not defined (can be anything).")],-1)),l(s,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[2]||(e[2]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4424025335aa82e086aee80dcf18b216e710e6b3/src/mlir/Dialects/TPU.jl#L136-L142",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",m,[t("summary",null,[e[5]||(e[5]=t("a",{id:"Reactant.MLIR.Dialects.tpu.create_subelement_mask-Tuple{}",href:"#Reactant.MLIR.Dialects.tpu.create_subelement_mask-Tuple{}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tpu.create_subelement_mask")],-1)),e[6]||(e[6]=a()),l(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[8]||(e[8]=c('<p><code>create_subelement_mask</code></p><p>The &quot;half-sublanes&quot;, &quot;quarter-sublanes&quot;, etc. (unit is determined by the type of <code>output</code>) of the mask are masked in the range specified by <code>from</code> and <code>to</code>.</p><ul><li><p>If <code>from &lt;= to</code>, the range <code>[from, to)</code> is set and the rest is unset.</p></li><li><p>If <code>to &lt;= from</code>, the range <code>[to, from)</code> is unset and the rest is set.</p></li></ul><p>All lanes are set identically.</p><p><strong>Example</strong></p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>%msk = tpu.create_subelement_mask 3, 9 : vector&lt;8x128x2xi1&gt;</span></span></code></pre></div><p>This creates a mask <code>%msk</code> where, for all <code>lane</code>s, <code>%msk[*][lane][*]</code> is:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>[[0, 0], [0, 1], [1, 1], [1, 1], [1, 0], [0, 0], [0, 0], [0, 0]]</span></span></code></pre></div><p>It is currently only supported:</p><ul><li><p>In TPU v4, for <code>num_subelems</code> of 1 and 2.</p></li><li><p>In TPU v5, for <code>num_subelems</code> of 1, 2, and 4.</p></li></ul>',10)),l(s,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[7]||(e[7]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4424025335aa82e086aee80dcf18b216e710e6b3/src/mlir/Dialects/TPU.jl#L204-L231",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})])])}const y=o(p,[["render",b]]);export{R as __pageData,y as default};
