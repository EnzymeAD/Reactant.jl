import{_ as T,C as i,c as o,o as r,j as t,a,G as n,a2 as s}from"./chunks/framework.C_VspMbV.js";const n1=JSON.parse('{"title":"Triton Dialect","description":"","frontmatter":{},"headers":[],"relativePath":"api/triton.md","filePath":"api/triton.md","lastUpdated":null}'),Q={name:"api/triton.md"},d={class:"jldocstring custom-block"},p={class:"jldocstring custom-block"},m={class:"jldocstring custom-block"},u={class:"jldocstring custom-block"},c={class:"jldocstring custom-block"},R={class:"jldocstring custom-block"},g={class:"jldocstring custom-block"},h={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},f={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"20.248ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 8949.4 1000","aria-hidden":"true"},L={class:"jldocstring custom-block"},b={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},M={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"26.094ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 11533.4 1000","aria-hidden":"true"},I={class:"jldocstring custom-block"},y={class:"jldocstring custom-block"},j={class:"jldocstring custom-block"},x={class:"jldocstring custom-block"},D={class:"jldocstring custom-block"},w={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},k={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"8.011ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 3541 1000","aria-hidden":"true"},V={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},H={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"8.011ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 3541 1000","aria-hidden":"true"},v={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},Z={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"7.778ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 3438 1000","aria-hidden":"true"},A={class:"jldocstring custom-block"},E={class:"jldocstring custom-block"},C={class:"jldocstring custom-block"},z={class:"jldocstring custom-block"},O={class:"jldocstring custom-block"},S={class:"jldocstring custom-block"},P={class:"jldocstring custom-block"},F={class:"jldocstring custom-block"},N={class:"jldocstring custom-block"},q={class:"jldocstring custom-block"},B={class:"jldocstring custom-block"},G={class:"jldocstring custom-block"},U={class:"jldocstring custom-block"},$={class:"jldocstring custom-block"},J={class:"jldocstring custom-block"},W={class:"jldocstring custom-block"},X={class:"jldocstring custom-block"};function _(Y,e,K,t1,e1,a1){const l=i("Badge");return r(),o("div",null,[e[172]||(e[172]=t("h1",{id:"Triton-Dialect",tabindex:"-1"},[a("Triton Dialect "),t("a",{class:"header-anchor",href:"#Triton-Dialect","aria-label":'Permalink to "Triton Dialect {#Triton-Dialect}"'},"​")],-1)),e[173]||(e[173]=t("p",null,[a("Refer to the "),t("a",{href:"https://triton-lang.org/main/dialects/TritonDialect.html",target:"_blank",rel:"noreferrer"},"official documentation"),a(" for more details.")],-1)),t("details",d,[t("summary",null,[e[0]||(e[0]=t("a",{id:"Reactant.MLIR.Dialects.tt.assert-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tt.assert-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.assert")],-1)),e[1]||(e[1]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[2]||(e[2]=t("p",null,[t("code",null,"assert")],-1)),e[3]||(e[3]=t("p",null,[t("code",null,"tt.assert"),a(" takes a condition tensor and a message string. If the condition is false, the message is printed, and the program is aborted.")],-1)),e[4]||(e[4]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L222-L227",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",p,[t("summary",null,[e[5]||(e[5]=t("a",{id:"Reactant.MLIR.Dialects.tt.atomic_cas-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tt.atomic_cas-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.atomic_cas")],-1)),e[6]||(e[6]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[7]||(e[7]=t("p",null,[t("code",null,"atomic_cas")],-1)),e[8]||(e[8]=t("p",null,"compare cmp with data old at location ptr,",-1)),e[9]||(e[9]=t("p",null,"if old == cmp, store val to ptr,",-1)),e[10]||(e[10]=t("p",null,"else store old to ptr,",-1)),e[11]||(e[11]=t("p",null,"return old",-1)),e[12]||(e[12]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L247-L257",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",m,[t("summary",null,[e[13]||(e[13]=t("a",{id:"Reactant.MLIR.Dialects.tt.atomic_rmw",href:"#Reactant.MLIR.Dialects.tt.atomic_rmw"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.atomic_rmw")],-1)),e[14]||(e[14]=a()),n(l,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[15]||(e[15]=t("p",null,[t("code",null,"atomic_rmw")],-1)),e[16]||(e[16]=t("p",null,"load data at ptr, do rmw_op with val, and store result to ptr.",-1)),e[17]||(e[17]=t("p",null,"return old value at ptr",-1)),e[18]||(e[18]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L279-L285",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",u,[t("summary",null,[e[19]||(e[19]=t("a",{id:"Reactant.MLIR.Dialects.tt.broadcast-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tt.broadcast-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.broadcast")],-1)),e[20]||(e[20]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[21]||(e[21]=t("p",null,[t("code",null,"broadcast")],-1)),e[22]||(e[22]=t("p",null,"For a given tensor, broadcast changes one or more dimensions with size 1 to a new size, e.g. tensor<1x32x1xf32> -> tensor<2x32x4xf32>. You cannot change the size of a non-1 dimension.",-1)),e[23]||(e[23]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L338-L344",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",c,[t("summary",null,[e[24]||(e[24]=t("a",{id:"Reactant.MLIR.Dialects.tt.call-Tuple{Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.tt.call-Tuple{Vector{Reactant.MLIR.IR.Value}}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.call")],-1)),e[25]||(e[25]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[26]||(e[26]=s("",5))]),t("details",R,[t("summary",null,[e[27]||(e[27]=t("a",{id:"Reactant.MLIR.Dialects.tt.clampf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tt.clampf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.clampf")],-1)),e[28]||(e[28]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[29]||(e[29]=t("p",null,[t("code",null,"clampf")],-1)),e[30]||(e[30]=t("p",null,"Clamp operation for floating point types.",-1)),e[31]||(e[31]=t("p",null,"The operation takes three arguments: x, min, and max. It returns a tensor of the same shape as x with its values clamped to the range [min, max].",-1)),e[32]||(e[32]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L383-L389",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",g,[t("summary",null,[e[33]||(e[33]=t("a",{id:"Reactant.MLIR.Dialects.tt.dot-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tt.dot-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.dot")],-1)),e[34]||(e[34]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[37]||(e[37]=t("p",null,[t("code",null,"dot")],-1)),t("mjx-container",h,[(r(),o("svg",f,e[35]||(e[35]=[s("",1)]))),e[36]||(e[36]=t("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[t("mi",null,"d"),t("mo",null,"="),t("mi",null,"m"),t("mi",null,"a"),t("mi",null,"t"),t("mi",null,"r"),t("mi",null,"i"),t("msub",null,[t("mi",null,"x"),t("mi",null,"m")]),t("mi",null,"u"),t("mi",null,"l"),t("mi",null,"t"),t("mi",null,"i"),t("mi",null,"p"),t("mi",null,"l"),t("mi",null,"y"),t("mo",{stretchy:"false"},"(")])],-1))]),e[38]||(e[38]=t("p",null,"a, b) + c. inputPrecision describes how to exercise the TC when the inputs are f32. It can be one of: tf32, tf32x3, ieee. tf32: use TC with tf32 ops. tf32x3: implement the 3xTF32 trick. For more info see the pass in F32DotTC.cpp ieee: don't use TC, implement dot in software. If the GPU does not have Tensor cores or the inputs are not f32, this flag is ignored.",-1)),e[39]||(e[39]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L417-L426",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",L,[t("summary",null,[e[40]||(e[40]=t("a",{id:"Reactant.MLIR.Dialects.tt.dot_scaled",href:"#Reactant.MLIR.Dialects.tt.dot_scaled"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.dot_scaled")],-1)),e[41]||(e[41]=a()),n(l,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),e[44]||(e[44]=t("p",null,[t("code",null,"dot_scaled")],-1)),t("mjx-container",b,[(r(),o("svg",M,e[42]||(e[42]=[s("",1)]))),e[43]||(e[43]=t("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[t("mi",null,"d"),t("mo",null,"="),t("mi",null,"m"),t("mi",null,"a"),t("mi",null,"t"),t("mi",null,"r"),t("mi",null,"i"),t("msub",null,[t("mi",null,"x"),t("mi",null,"m")]),t("mi",null,"u"),t("mi",null,"l"),t("mi",null,"t"),t("mi",null,"i"),t("mi",null,"p"),t("mi",null,"l"),t("mi",null,"y"),t("mo",{stretchy:"false"},"("),t("mi",null,"s"),t("mi",null,"c"),t("mi",null,"a"),t("mi",null,"l"),t("mi",null,"e"),t("mo",{stretchy:"false"},"(")])],-1))]),e[45]||(e[45]=t("p",null,"lhs, lhs_scale), scale(rlhs, rhs_scale)) + c. Where scale(x, s) is a function that applies the scale per block following microscaling spec.",-1)),e[46]||(e[46]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L459-L464",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",I,[t("summary",null,[e[47]||(e[47]=t("a",{id:"Reactant.MLIR.Dialects.tt.elementwise_inline_asm-Tuple{Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.tt.elementwise_inline_asm-Tuple{Vector{Reactant.MLIR.IR.Value}}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.elementwise_inline_asm")],-1)),e[48]||(e[48]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[49]||(e[49]=t("p",null,[t("code",null,"elementwise_inline_asm")],-1)),e[50]||(e[50]=t("p",null,"Runs an inline asm block to generate one or more tensors.",-1)),e[51]||(e[51]=t("p",null,[a("The asm block is given "),t("code",null,"packed_element"),a(" elements at a time. Exactly which elems it receives is unspecified.")],-1)),e[52]||(e[52]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L513-L520",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",y,[t("summary",null,[e[53]||(e[53]=t("a",{id:"Reactant.MLIR.Dialects.tt.experimental_descriptor_gather-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tt.experimental_descriptor_gather-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.experimental_descriptor_gather")],-1)),e[54]||(e[54]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[55]||(e[55]=t("p",null,[t("code",null,"experimental_descriptor_gather")],-1)),e[56]||(e[56]=t("p",null,[a("The "),t("code",null,"tt.experimental_desciptor_gather"),a(" op will be lowered to NVIDIA TMA load operations on targets that support it.")],-1)),e[57]||(e[57]=t("p",null,[t("code",null,"desc_ptr"),a(" is a pointer to the TMA descriptor allocated in global memory. The descriptor block must have 1 row and the indices must be a 1D tensor. Accordingly, the result is a 2D tensor multiple rows.")],-1)),e[58]||(e[58]=t("p",null,"This is an escape hatch and is only there for testing/experimenting. This op will be removed in the future.",-1)),e[59]||(e[59]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L575-L587",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",j,[t("summary",null,[e[60]||(e[60]=t("a",{id:"Reactant.MLIR.Dialects.tt.experimental_descriptor_load-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.tt.experimental_descriptor_load-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.experimental_descriptor_load")],-1)),e[61]||(e[61]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[62]||(e[62]=t("p",null,[t("code",null,"experimental_descriptor_load")],-1)),e[63]||(e[63]=t("p",null,[a("This operation will be lowered to Nvidia TMA load operation on targets supporting it. "),t("code",null,"desc"),a(" is a tensor descriptor object. The destination tensor type and shape must match the descriptor otherwise the result is undefined.")],-1)),e[64]||(e[64]=t("p",null,"This is an escape hatch and is only there for testing/experimenting. This op will be removed in the future.",-1)),e[65]||(e[65]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L609-L618",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",x,[t("summary",null,[e[66]||(e[66]=t("a",{id:"Reactant.MLIR.Dialects.tt.experimental_descriptor_store-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.tt.experimental_descriptor_store-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.experimental_descriptor_store")],-1)),e[67]||(e[67]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[68]||(e[68]=t("p",null,[t("code",null,"experimental_descriptor_store")],-1)),e[69]||(e[69]=t("p",null,[a("This operation will be lowered to Nvidia TMA store operation on targets supporting it. "),t("code",null,"desc"),a(" is a tensor descriptor object. The shape and types of "),t("code",null,"src"),a(" must match the descriptor otherwise the result is undefined.")],-1)),e[70]||(e[70]=t("p",null,"This is an escape hatch and is only there for testing/experimenting. This op will be removed in the future.",-1)),e[71]||(e[71]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L668-L677",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",D,[t("summary",null,[e[72]||(e[72]=t("a",{id:"Reactant.MLIR.Dialects.tt.extern_elementwise-Tuple{Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.tt.extern_elementwise-Tuple{Vector{Reactant.MLIR.IR.Value}}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.extern_elementwise")],-1)),e[73]||(e[73]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[84]||(e[84]=t("p",null,[t("code",null,"extern_elementwise")],-1)),t("p",null,[e[80]||(e[80]=a("call an external function $symbol implemented in ")),t("mjx-container",w,[(r(),o("svg",k,e[74]||(e[74]=[s("",1)]))),e[75]||(e[75]=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"l"),t("mi",null,"i"),t("mi",null,"b"),t("mi",null,"p"),t("mi",null,"a"),t("mi",null,"t"),t("mi",null,"h"),t("mrow",{"data-mjx-texclass":"ORD"},[t("mo",null,"/")])])],-1))]),e[81]||(e[81]=a("libname with $args return ")),t("mjx-container",V,[(r(),o("svg",H,e[76]||(e[76]=[s("",1)]))),e[77]||(e[77]=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"l"),t("mi",null,"i"),t("mi",null,"b"),t("mi",null,"p"),t("mi",null,"a"),t("mi",null,"t"),t("mi",null,"h"),t("mrow",{"data-mjx-texclass":"ORD"},[t("mo",null,"/")])])],-1))]),e[82]||(e[82]=a("libname:")),t("mjx-container",v,[(r(),o("svg",Z,e[78]||(e[78]=[s("",1)]))),e[79]||(e[79]=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mi",null,"s"),t("mi",null,"y"),t("mi",null,"m"),t("mi",null,"b"),t("mi",null,"o"),t("mi",null,"l"),t("mo",{stretchy:"false"},"(")])],-1))]),e[83]||(e[83]=a("args...)"))]),e[85]||(e[85]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L772-L777",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",A,[t("summary",null,[e[86]||(e[86]=t("a",{id:"Reactant.MLIR.Dialects.tt.fp_to_fp-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tt.fp_to_fp-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.fp_to_fp")],-1)),e[87]||(e[87]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[88]||(e[88]=t("p",null,[t("code",null,"fp_to_fp")],-1)),e[89]||(e[89]=t("p",null,"Floating point casting for custom types (F8), and non-default rounding modes.",-1)),e[90]||(e[90]=t("p",null,"F8 <-> FP16, BF16, FP32, FP64",-1)),e[91]||(e[91]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L810-L816",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",E,[t("summary",null,[e[92]||(e[92]=t("a",{id:"Reactant.MLIR.Dialects.tt.func-Tuple{}",href:"#Reactant.MLIR.Dialects.tt.func-Tuple{}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.func")],-1)),e[93]||(e[93]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[94]||(e[94]=s("",6))]),t("details",C,[t("summary",null,[e[95]||(e[95]=t("a",{id:"Reactant.MLIR.Dialects.tt.gather-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tt.gather-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.gather")],-1)),e[96]||(e[96]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[97]||(e[97]=t("p",null,[t("code",null,"gather")],-1)),e[98]||(e[98]=t("p",null,"Gather elements from the input tensor using the indices tensor along a single specified axis. The output tensor has the same shape as the indices tensor. The input and indices tensors must have the same number of dimension, and each dimension of the indices tensor that is not the gather dimension cannot be greater than the corresponding dimension in the input tensor.",-1)),e[99]||(e[99]=t("p",null,[a("The "),t("code",null,"efficient_layout"),a(" attribute is set when the compiler has determined an optimized layout for the operation, indicating that it should not be changed.")],-1)),e[100]||(e[100]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L837-L850",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",z,[t("summary",null,[e[101]||(e[101]=t("a",{id:"Reactant.MLIR.Dialects.tt.histogram-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tt.histogram-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.histogram")],-1)),e[102]||(e[102]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[103]||(e[103]=t("p",null,[t("code",null,"histogram")],-1)),e[104]||(e[104]=t("p",null,"Return the histogram of the input tensor. The number of bins is equal to the dimension of the output tensor. Each bins has a width of 1 and bins start at 0.",-1)),e[105]||(e[105]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L922-L928",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",O,[t("summary",null,[e[106]||(e[106]=t("a",{id:"Reactant.MLIR.Dialects.tt.join-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tt.join-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.join")],-1)),e[107]||(e[107]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[108]||(e[108]=t("p",null,[t("code",null,"join")],-1)),e[109]||(e[109]=t("p",null,"For example, if the two input tensors are 4x8xf32, returns a tensor of shape 4x8x2xf32.",-1)),e[110]||(e[110]=t("p",null,"Because Triton tensors always have a power-of-two number of elements, the two input tensors must have the same shape.",-1)),e[111]||(e[111]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L967-L975",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",S,[t("summary",null,[e[112]||(e[112]=t("a",{id:"Reactant.MLIR.Dialects.tt.make_range-Tuple{}",href:"#Reactant.MLIR.Dialects.tt.make_range-Tuple{}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.make_range")],-1)),e[113]||(e[113]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[114]||(e[114]=t("p",null,[t("code",null,"make_range")],-1)),e[115]||(e[115]=t("p",null,"Returns an 1D int32 tensor.",-1)),e[116]||(e[116]=t("p",null,"Values span from start to $end (exclusive), with step = 1",-1)),e[117]||(e[117]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L1047-L1053",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",P,[t("summary",null,[e[118]||(e[118]=t("a",{id:"Reactant.MLIR.Dialects.tt.make_tensor_descriptor-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.tt.make_tensor_descriptor-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.make_tensor_descriptor")],-1)),e[119]||(e[119]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[120]||(e[120]=t("p",null,[t("code",null,"make_tensor_descriptor")],-1)),e[121]||(e[121]=t("p",null,[t("code",null,"tt.make_tensor_descriptor"),a(" takes both meta information of the parent tensor and the block size, and returns a descriptor object which can be used to load/store from the tensor in global memory.")],-1)),e[122]||(e[122]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L1073-L1078",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",F,[t("summary",null,[e[123]||(e[123]=t("a",{id:"Reactant.MLIR.Dialects.tt.make_tensor_ptr-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.tt.make_tensor_ptr-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.make_tensor_ptr")],-1)),e[124]||(e[124]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[125]||(e[125]=t("p",null,[t("code",null,"make_tensor_ptr")],-1)),e[126]||(e[126]=t("p",null,[t("code",null,"tt.make_tensor_ptr"),a(" takes both meta information of the parent tensor and the block tensor, then it returns a pointer to the block tensor, e.g. returns a type of "),t("code",null,"tt.ptr<tensor<8x8xf16>>"),a(".")],-1)),e[127]||(e[127]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L1104-L1109",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",N,[t("summary",null,[e[128]||(e[128]=t("a",{id:"Reactant.MLIR.Dialects.tt.mulhiui-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tt.mulhiui-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.mulhiui")],-1)),e[129]||(e[129]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[130]||(e[130]=t("p",null,[t("code",null,"mulhiui")],-1)),e[131]||(e[131]=t("p",null,"Most significant N bits of the 2N-bit product of two integers.",-1)),e[132]||(e[132]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L1137-L1141",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",q,[t("summary",null,[e[133]||(e[133]=t("a",{id:"Reactant.MLIR.Dialects.tt.precise_divf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tt.precise_divf-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.precise_divf")],-1)),e[134]||(e[134]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[135]||(e[135]=t("p",null,[t("code",null,"precise_divf")],-1)),e[136]||(e[136]=t("p",null,"Precise div for floating point types.",-1)),e[137]||(e[137]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L1164-L1168",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",B,[t("summary",null,[e[138]||(e[138]=t("a",{id:"Reactant.MLIR.Dialects.tt.precise_sqrt-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tt.precise_sqrt-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.precise_sqrt")],-1)),e[139]||(e[139]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[140]||(e[140]=t("p",null,[t("code",null,"precise_sqrt")],-1)),e[141]||(e[141]=t("p",null,"Precise sqrt for floating point types.",-1)),e[142]||(e[142]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L1191-L1195",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",G,[t("summary",null,[e[143]||(e[143]=t("a",{id:"Reactant.MLIR.Dialects.tt.print-Tuple{Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.tt.print-Tuple{Vector{Reactant.MLIR.IR.Value}}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.print")],-1)),e[144]||(e[144]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[145]||(e[145]=t("p",null,[t("code",null,"print")],-1)),e[146]||(e[146]=t("p",null,[t("code",null,"tt.print"),a(" takes a literal string prefix and an arbitrary number of scalar or tensor arguments that should be printed. format are generated automatically from the arguments.")],-1)),e[147]||(e[147]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L1216-L1221",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",U,[t("summary",null,[e[148]||(e[148]=t("a",{id:"Reactant.MLIR.Dialects.tt.reinterpret_tensor_descriptor-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tt.reinterpret_tensor_descriptor-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.reinterpret_tensor_descriptor")],-1)),e[149]||(e[149]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[150]||(e[150]=t("p",null,[t("code",null,"reinterpret_tensor_descriptor")],-1)),e[151]||(e[151]=t("p",null,"This Op exists to help the transition from untyped raw TMA objects to typed Tensor descriptor objects. Ideally, we can remove this once the APIs are fully fleshed out.",-1)),e[152]||(e[152]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L123-L128",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",$,[t("summary",null,[e[153]||(e[153]=t("a",{id:"Reactant.MLIR.Dialects.tt.reshape-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tt.reshape-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.reshape")],-1)),e[154]||(e[154]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[155]||(e[155]=t("p",null,[t("code",null,"reshape")],-1)),e[156]||(e[156]=t("p",null,"reinterpret a tensor to a different shape.",-1)),e[157]||(e[157]=t("p",null,"If allow_reorder is set the compiler is free to change the order of elements to generate more efficient code.",-1)),e[158]||(e[158]=t("p",null,"If efficient_layout is set, this is a hint that the destination layout should be kept for performance reason. The compiler is still free to change it for better performance.",-1)),e[159]||(e[159]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L1308-L1318",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",J,[t("summary",null,[e[160]||(e[160]=t("a",{id:"Reactant.MLIR.Dialects.tt.return_-Tuple{Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.tt.return_-Tuple{Vector{Reactant.MLIR.IR.Value}}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.return_")],-1)),e[161]||(e[161]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[162]||(e[162]=s("",5))]),t("details",W,[t("summary",null,[e[163]||(e[163]=t("a",{id:"Reactant.MLIR.Dialects.tt.split-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tt.split-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.split")],-1)),e[164]||(e[164]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[165]||(e[165]=t("p",null,[t("code",null,"split")],-1)),e[166]||(e[166]=t("p",null,"The input must be a tensor whose last dimension has size 2. Returns two tensors, src[..., 0] and src[..., 1].",-1)),e[167]||(e[167]=t("p",null,"For example, if the input shape is 4x8x2xf32, returns two tensors of shape 4x8xf32.",-1)),e[168]||(e[168]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/3ce001c838adace3118304d55ce0e6c041f53554/src/mlir/Dialects/Triton.jl#L1414-L1422",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",X,[t("summary",null,[e[169]||(e[169]=t("a",{id:"Reactant.MLIR.Dialects.tt.trans-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.tt.trans-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.tt.trans")],-1)),e[170]||(e[170]=a()),n(l,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[171]||(e[171]=s("",9))])])}const s1=T(Q,[["render",_]]);export{n1 as __pageData,s1 as default};
