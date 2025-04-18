import{_ as o,C as i,c as d,o as r,j as t,a,G as l,w as n,a2 as c}from"./chunks/framework.5PQhGlTR.js";const R=JSON.parse('{"title":"TPU Dialect","description":"","frontmatter":{},"headers":[],"relativePath":"api/dialects/tpu.md","filePath":"api/dialects/tpu.md","lastUpdated":null}'),p={name:"api/dialects/tpu.md"},u={class:"jldocstring custom-block"},m={class:"jldocstring custom-block"};function b(f,e,h,g,_,k){const s=i("Badge");return r(),d("div",null,[e[9]||(e[9]=t("h1",{id:"TPU-Dialect",tabindex:"-1"},[a("TPU Dialect "),t("a",{class:"header-anchor",href:"#TPU-Dialect","aria-label":'Permalink to "TPU Dialect {#TPU-Dialect}"'},"​")],-1)),e[10]||(e[10]=t("p",null,[a("Refer to the "),t("a",{href:"https://github.com/jax-ml/jax/blob/main/jaxlib/mosaic/dialect/tpu/tpu.td",target:"_blank",rel:"noreferrer"},"official documentation"),a(" for more details.")],-1)),t("details",u,[t("summary",null,[e[0]||(e[0]=t("a",{id:"Reactant.MLIR.Dialects.tpu.broadcast_in_sublanes-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tpu.broadcast_in_sublanes-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tpu.broadcast_in_sublanes")],-1)),e[1]||(e[1]=a()),l(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[3]||(e[3]=t("p",null,[t("code",null,"broadcast_in_sublanes")],-1)),e[4]||(e[4]=t("p",null,[a("For each sublane "),t("code",null,"i"),a(", broadcasts the value in lane "),t("code",null,"lane + i"),a(" along the entire sublane. If "),t("code",null,"lane + i"),a(" is not in [0, lane_count), then the value in sublane "),t("code",null,"i"),a(" is not defined (can be anything).")],-1)),l(s,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[2]||(e[2]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a5a7e53f93d3589043dc75507557cd0adf8b6d05/src/mlir/Dialects/TPU.jl#L136-L142",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",m,[t("summary",null,[e[5]||(e[5]=t("a",{id:"Reactant.MLIR.Dialects.tpu.create_subelement_mask-Tuple{}",href:"#Reactant.MLIR.Dialects.tpu.create_subelement_mask-Tuple{}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tpu.create_subelement_mask")],-1)),e[6]||(e[6]=a()),l(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[8]||(e[8]=c("",10)),l(s,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[7]||(e[7]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a5a7e53f93d3589043dc75507557cd0adf8b6d05/src/mlir/Dialects/TPU.jl#L204-L231",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})])])}const y=o(p,[["render",b]]);export{R as __pageData,y as default};
