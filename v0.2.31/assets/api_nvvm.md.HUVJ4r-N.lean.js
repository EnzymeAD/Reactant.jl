import{_ as i,C as d,c as r,o,j as e,a,G as s,a2 as l}from"./chunks/framework.B5OvDpLa.js";const ee=JSON.parse('{"title":"NVVM Dialect","description":"","frontmatter":{},"headers":[],"relativePath":"api/nvvm.md","filePath":"api/nvvm.md","lastUpdated":null}'),c={name:"api/nvvm.md"},p={class:"jldocstring custom-block"},m={class:"jldocstring custom-block"},T={class:"jldocstring custom-block"},u={class:"jldocstring custom-block"},Q={class:"jldocstring custom-block"},h={class:"jldocstring custom-block"},b={class:"jldocstring custom-block"},f={class:"jldocstring custom-block"},g={class:"jldocstring custom-block"},v={class:"jldocstring custom-block"},y={class:"jldocstring custom-block"},k={class:"jldocstring custom-block"},R={class:"jldocstring custom-block"},_={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},L={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.464ex"},xmlns:"http://www.w3.org/2000/svg",width:"53.79ex",height:"2.059ex",role:"img",focusable:"false",viewBox:"0 -705 23775.1 910","aria-hidden":"true"},M={class:"jldocstring custom-block"},x={class:"jldocstring custom-block"},I={class:"jldocstring custom-block"},D={class:"jldocstring custom-block"},j={class:"jldocstring custom-block"},w={class:"jldocstring custom-block"},A={class:"jldocstring custom-block"},V={class:"jldocstring custom-block"},S={class:"jldocstring custom-block"},H={class:"jldocstring custom-block"},C={class:"jldocstring custom-block"},P={class:"jldocstring custom-block"},N={class:"jldocstring custom-block"},F={class:"jldocstring custom-block"},Z={class:"jldocstring custom-block"},O={class:"jldocstring custom-block"},E={class:"jldocstring custom-block"},z={class:"jldocstring custom-block"},X={class:"jldocstring custom-block"},q={class:"jldocstring custom-block"},B={class:"jldocstring custom-block"};function $(W,t,G,J,K,U){const n=d("Badge");return o(),r("div",null,[t[166]||(t[166]=e("h1",{id:"NVVM-Dialect",tabindex:"-1"},[a("NVVM Dialect "),e("a",{class:"header-anchor",href:"#NVVM-Dialect","aria-label":'Permalink to "NVVM Dialect {#NVVM-Dialect}"'},"​")],-1)),t[167]||(t[167]=e("p",null,[a("Refer to the "),e("a",{href:"https://mlir.llvm.org/docs/Dialects/NVVMDialect/",target:"_blank",rel:"noreferrer"},"official documentation"),a(" for more details.")],-1)),e("details",p,[e("summary",null,[t[0]||(t[0]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.barrier_arrive",href:"#Reactant.MLIR.Dialects.nvvm.barrier_arrive"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.barrier_arrive")],-1)),t[1]||(t[1]=a()),s(n,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),t[2]||(t[2]=e("p",null,[e("code",null,"barrier_arrive")],-1)),t[3]||(t[3]=e("p",null,"Thread that executes this op announces their arrival at the barrier with given id and continue their execution.",-1)),t[4]||(t[4]=e("p",null,[a("The default barrier id is 0 that is similar to "),e("code",null,"nvvm.barrier"),a(" Op. When "),e("code",null,"barrierId"),a(" is not present, the default barrier id is used.")],-1)),t[5]||(t[5]=e("p",null,[e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar",target:"_blank",rel:"noreferrer"},"For more information, see PTX ISA")],-1)),t[6]||(t[6]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L35-L45",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",m,[e("summary",null,[t[7]||(t[7]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.breakpoint-Tuple{}",href:"#Reactant.MLIR.Dialects.nvvm.breakpoint-Tuple{}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.breakpoint")],-1)),t[8]||(t[8]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[9]||(t[9]=e("p",null,[e("code",null,"breakpoint")],-1)),t[10]||(t[10]=e("p",null,[a("Breakpoint suspends execution of the program for debugging. "),e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-brkpt",target:"_blank",rel:"noreferrer"},"For more information, see PTX ISA")],-1)),t[11]||(t[11]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L282-L287",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",T,[e("summary",null,[t[12]||(t[12]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.cluster_arrive-Tuple{}",href:"#Reactant.MLIR.Dialects.nvvm.cluster_arrive-Tuple{}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.cluster_arrive")],-1)),t[13]||(t[13]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[14]||(t[14]=l("",5))]),e("details",u,[e("summary",null,[t[15]||(t[15]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.cluster_arrive_relaxed-Tuple{}",href:"#Reactant.MLIR.Dialects.nvvm.cluster_arrive_relaxed-Tuple{}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.cluster_arrive_relaxed")],-1)),t[16]||(t[16]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[17]||(t[17]=l("",5))]),e("details",Q,[e("summary",null,[t[18]||(t[18]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.cluster_wait-Tuple{}",href:"#Reactant.MLIR.Dialects.nvvm.cluster_wait-Tuple{}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.cluster_wait")],-1)),t[19]||(t[19]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[20]||(t[20]=l("",4))]),e("details",h,[e("summary",null,[t[21]||(t[21]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.cp_async_bulk_commit_group-Tuple{}",href:"#Reactant.MLIR.Dialects.nvvm.cp_async_bulk_commit_group-Tuple{}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.cp_async_bulk_commit_group")],-1)),t[22]||(t[22]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[23]||(t[23]=e("p",null,[e("code",null,"cp_async_bulk_commit_group")],-1)),t[24]||(t[24]=e("p",null,"This Op commits all prior initiated but uncommitted cp.async.bulk instructions into a cp.async.bulk-group.",-1)),t[25]||(t[25]=e("p",null,[e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group",target:"_blank",rel:"noreferrer"},"For more information, see PTX ISA")],-1)),t[26]||(t[26]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L659-L666",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",b,[e("summary",null,[t[27]||(t[27]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.cp_async_bulk_global_shared_cta",href:"#Reactant.MLIR.Dialects.nvvm.cp_async_bulk_global_shared_cta"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.cp_async_bulk_global_shared_cta")],-1)),t[28]||(t[28]=a()),s(n,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),t[29]||(t[29]=e("p",null,[e("code",null,"cp_async_bulk_global_shared_cta")],-1)),t[30]||(t[30]=e("p",null,"Initiates an asynchronous copy operation from Shared CTA memory to global memory.",-1)),t[31]||(t[31]=e("p",null,[a("The "),e("code",null,"l2CacheHint"),a(" operand is optional, and it is used to specify cache eviction policy that may be used during the memory access.")],-1)),t[32]||(t[32]=e("p",null,[e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk",target:"_blank",rel:"noreferrer"},"For more information, see PTX ISA")],-1)),t[33]||(t[33]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L745-L755",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",f,[e("summary",null,[t[34]||(t[34]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.cp_async_bulk_shared_cluster_global",href:"#Reactant.MLIR.Dialects.nvvm.cp_async_bulk_shared_cluster_global"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.cp_async_bulk_shared_cluster_global")],-1)),t[35]||(t[35]=a()),s(n,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),t[36]||(t[36]=l("",6))]),e("details",g,[e("summary",null,[t[37]||(t[37]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.cp_async_bulk_shared_cluster_shared_cta-NTuple{4, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.nvvm.cp_async_bulk_shared_cluster_shared_cta-NTuple{4, Reactant.MLIR.IR.Value}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.cp_async_bulk_shared_cluster_shared_cta")],-1)),t[38]||(t[38]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[39]||(t[39]=e("p",null,[e("code",null,"cp_async_bulk_shared_cluster_shared_cta")],-1)),t[40]||(t[40]=e("p",null,"Initiates an asynchronous copy operation from Shared CTA memory to Shared cluster memory.",-1)),t[41]||(t[41]=e("p",null,[e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk",target:"_blank",rel:"noreferrer"},"For more information, see PTX ISA")],-1)),t[42]||(t[42]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L782-L789",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",v,[e("summary",null,[t[43]||(t[43]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_prefetch",href:"#Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_prefetch"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_prefetch")],-1)),t[44]||(t[44]=a()),s(n,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),t[45]||(t[45]=l("",10))]),e("details",y,[e("summary",null,[t[46]||(t[46]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_reduce",href:"#Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_reduce"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_reduce")],-1)),t[47]||(t[47]=a()),s(n,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),t[48]||(t[48]=e("p",null,[e("code",null,"cp_async_bulk_tensor_reduce")],-1)),t[49]||(t[49]=e("p",null,"Initiates an asynchronous reduction operation of tensor data in global memory with tensor data in shared memory.",-1)),t[50]||(t[50]=e("p",{"add,":"","min,":"","max,":"","inc,":"","dec,":"","and,":"","or,":"",xor:""},[a("The "),e("code",null,"mode"),a(" attribute indicates whether the copy mode is tile or im2col. The "),e("code",null,"redOp"),a(" attribute specifies the reduction operations applied. The supported reduction operations are:")],-1)),t[51]||(t[51]=e("p",null,[a("The "),e("code",null,"l2CacheHint"),a(" operand is optional, and it is used to specify cache eviction policy that may be used during the memory access.")],-1)),t[52]||(t[52]=e("p",null,[e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor",target:"_blank",rel:"noreferrer"},"For more information, see PTX ISA")],-1)),t[53]||(t[53]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L940-L955",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",k,[e("summary",null,[t[54]||(t[54]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_shared_cluster_global",href:"#Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_shared_cluster_global"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.cp_async_bulk_tensor_shared_cluster_global")],-1)),t[55]||(t[55]=a()),s(n,{type:"info",class:"jlObjectType jlFunction",text:"Function"})]),t[56]||(t[56]=l("",11))]),e("details",R,[e("summary",null,[t[57]||(t[57]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.cp_async_bulk_wait_group-Tuple{}",href:"#Reactant.MLIR.Dialects.nvvm.cp_async_bulk_wait_group-Tuple{}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.cp_async_bulk_wait_group")],-1)),t[58]||(t[58]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[65]||(t[65]=e("p",null,[e("code",null,"cp_async_bulk_wait_group")],-1)),t[66]||(t[66]=e("p",null,"Op waits for completion of the most recent bulk async-groups.",-1)),e("p",null,[t[61]||(t[61]=a("The ")),t[62]||(t[62]=e("code",null,"$group",-1)),t[63]||(t[63]=a(" operand tells waiting has to be done until for ")),e("mjx-container",_,[(o(),r("svg",L,t[59]||(t[59]=[l("",1)]))),t[60]||(t[60]=e("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[e("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[e("mi",null,"g"),e("mi",null,"r"),e("mi",null,"o"),e("mi",null,"u"),e("mi",null,"p"),e("mi",null,"o"),e("mi",null,"r"),e("mi",null,"f"),e("mi",null,"e"),e("mi",null,"w"),e("mi",null,"e"),e("mi",null,"r"),e("mi",null,"o"),e("mi",null,"f"),e("mi",null,"t"),e("mi",null,"h"),e("mi",null,"e"),e("mi",null,"m"),e("mi",null,"o"),e("mi",null,"s"),e("mi",null,"t"),e("mi",null,"r"),e("mi",null,"e"),e("mi",null,"c"),e("mi",null,"e"),e("mi",null,"n"),e("mi",null,"t"),e("mi",null,"b"),e("mi",null,"u"),e("mi",null,"l"),e("mi",null,"k"),e("mi",null,"a"),e("mi",null,"s"),e("mi",null,"y"),e("mi",null,"n"),e("mi",null,"c"),e("mo",null,"−"),e("mi",null,"g"),e("mi",null,"r"),e("mi",null,"o"),e("mi",null,"u"),e("mi",null,"p"),e("mi",null,"s"),e("mo",null,"."),e("mi",null,"I"),e("mi",null,"f"),e("mo",{"data-mjx-pseudoscript":"true"},"‘")])],-1))]),t[64]||(t[64]=a("group` is 0, the op wait until all the most recent bulk async-groups have completed."))]),t[67]||(t[67]=e("p",null,[a("The "),e("code",null,"$read"),a(" indicates that the waiting has to be done until all the bulk async operations in the specified bulk async-group have completed reading from their source locations.")],-1)),t[68]||(t[68]=e("p",null,[e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group",target:"_blank",rel:"noreferrer"},"For more information, see PTX ISA")],-1)),t[69]||(t[69]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L1019-L1033",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",M,[e("summary",null,[t[70]||(t[70]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.cp_async_mbarrier_arrive-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.nvvm.cp_async_mbarrier_arrive-Tuple{Reactant.MLIR.IR.Value}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.cp_async_mbarrier_arrive")],-1)),t[71]||(t[71]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[72]||(t[72]=l("",4))]),e("details",x,[e("summary",null,[t[73]||(t[73]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.cp_async_mbarrier_arrive_shared-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.nvvm.cp_async_mbarrier_arrive_shared-Tuple{Reactant.MLIR.IR.Value}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.cp_async_mbarrier_arrive_shared")],-1)),t[74]||(t[74]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[75]||(t[75]=l("",4))]),e("details",I,[e("summary",null,[t[76]||(t[76]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.cvt_float_to_tf32-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.nvvm.cvt_float_to_tf32-Tuple{Reactant.MLIR.IR.Value}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.cvt_float_to_tf32")],-1)),t[77]||(t[77]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[78]||(t[78]=l("",4))]),e("details",D,[e("summary",null,[t[79]||(t[79]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.elect_sync-Tuple{}",href:"#Reactant.MLIR.Dialects.nvvm.elect_sync-Tuple{}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.elect_sync")],-1)),t[80]||(t[80]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[81]||(t[81]=l("",4))]),e("details",j,[e("summary",null,[t[82]||(t[82]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.exit-Tuple{}",href:"#Reactant.MLIR.Dialects.nvvm.exit-Tuple{}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.exit")],-1)),t[83]||(t[83]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[84]||(t[84]=e("p",null,[e("code",null,"exit")],-1)),t[85]||(t[85]=e("p",null,[a("Ends execution of a thread. "),e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-exit",target:"_blank",rel:"noreferrer"},"For more information, see PTX ISA")],-1)),t[86]||(t[86]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L1856-L1861",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",w,[e("summary",null,[t[87]||(t[87]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.fence_mbarrier_init-Tuple{}",href:"#Reactant.MLIR.Dialects.nvvm.fence_mbarrier_init-Tuple{}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.fence_mbarrier_init")],-1)),t[88]||(t[88]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[89]||(t[89]=e("p",null,[e("code",null,"fence_mbarrier_init")],-1)),t[90]||(t[90]=e("p",null,"Fence operation that applies on the prior nvvm.mbarrier.init",-1)),t[91]||(t[91]=e("p",null,[e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar",target:"_blank",rel:"noreferrer"},"For more information, see PTX ISA")],-1)),t[92]||(t[92]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L1881-L1887",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",A,[e("summary",null,[t[93]||(t[93]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.fence_proxy-Tuple{}",href:"#Reactant.MLIR.Dialects.nvvm.fence_proxy-Tuple{}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.fence_proxy")],-1)),t[94]||(t[94]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[95]||(t[95]=e("p",null,[e("code",null,"fence_proxy")],-1)),t[96]||(t[96]=e("p",null,"Fence operation with proxy to establish an ordering between memory accesses that may happen through different proxies.",-1)),t[97]||(t[97]=e("p",null,[e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar",target:"_blank",rel:"noreferrer"},"For more information, see PTX ISA")],-1)),t[98]||(t[98]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L1946-L1953",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",V,[e("summary",null,[t[99]||(t[99]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.fence_proxy_acquire-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.nvvm.fence_proxy_acquire-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.fence_proxy_acquire")],-1)),t[100]||(t[100]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[101]||(t[101]=l("",5))]),e("details",S,[e("summary",null,[t[102]||(t[102]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.fence_proxy_release-Tuple{}",href:"#Reactant.MLIR.Dialects.nvvm.fence_proxy_release-Tuple{}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.fence_proxy_release")],-1)),t[103]||(t[103]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[104]||(t[104]=e("p",null,[e("code",null,"fence_proxy_release")],-1)),t[105]||(t[105]=e("p",null,[e("code",null,"fence.proxy.release"),a(" is a uni-directional fence used to establish ordering between a prior memory access performed via the generic proxy and a subsequent memory access performed via the tensormap proxy. "),e("code",null,"fence.proxy.release"),a(" operation can form a release sequence that synchronizes with an acquire sequence that contains the fence.proxy.acquire proxy fence operation")],-1)),t[106]||(t[106]=e("p",null,[e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar",target:"_blank",rel:"noreferrer"},"For more information, see PTX ISA")],-1)),t[107]||(t[107]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L1974-L1984",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",H,[e("summary",null,[t[108]||(t[108]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.griddepcontrol_launch_dependents-Tuple{}",href:"#Reactant.MLIR.Dialects.nvvm.griddepcontrol_launch_dependents-Tuple{}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.griddepcontrol_launch_dependents")],-1)),t[109]||(t[109]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[110]||(t[110]=e("p",null,[e("code",null,"griddepcontrol_launch_dependents")],-1)),t[111]||(t[111]=e("p",null,"Signals that specific dependents the runtime system designated to react to this instruction can be scheduled as soon as all other CTAs in the grid issue the same instruction or have completed.",-1)),t[112]||(t[112]=e("p",null,[e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol",target:"_blank",rel:"noreferrer"},"For more information, see PTX ISA")],-1)),t[113]||(t[113]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L2126-L2135",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",C,[e("summary",null,[t[114]||(t[114]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.griddepcontrol_wait-Tuple{}",href:"#Reactant.MLIR.Dialects.nvvm.griddepcontrol_wait-Tuple{}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.griddepcontrol_wait")],-1)),t[115]||(t[115]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[116]||(t[116]=e("p",null,[e("code",null,"griddepcontrol_wait")],-1)),t[117]||(t[117]=e("p",null,"Causes the executing thread to wait until all prerequisite grids in flight have completed and all the memory operations from the prerequisite grids are performed and made visible to the current grid.",-1)),t[118]||(t[118]=e("p",null,[e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol",target:"_blank",rel:"noreferrer"},"For more information, see PTX ISA")],-1)),t[119]||(t[119]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L2155-L2164",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",P,[e("summary",null,[t[120]||(t[120]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.mma_sync-Tuple{Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.nvvm.mma_sync-Tuple{Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}, Vector{Reactant.MLIR.IR.Value}}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.mma_sync")],-1)),t[121]||(t[121]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[122]||(t[122]=l("",12))]),e("details",N,[e("summary",null,[t[123]||(t[123]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.shfl_sync-NTuple{4, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.nvvm.shfl_sync-NTuple{4, Reactant.MLIR.IR.Value}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.shfl_sync")],-1)),t[124]||(t[124]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[125]||(t[125]=l("",4))]),e("details",F,[e("summary",null,[t[126]||(t[126]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.stmatrix-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.nvvm.stmatrix-Tuple{Reactant.MLIR.IR.Value, Vector{Reactant.MLIR.IR.Value}}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.stmatrix")],-1)),t[127]||(t[127]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[128]||(t[128]=e("p",null,[e("code",null,"stmatrix")],-1)),t[129]||(t[129]=e("p",null,"Collectively store one or more matrices across all threads in a warp to the location indicated by the address operand ptr in shared memory.",-1)),t[130]||(t[130]=e("p",null,[e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix",target:"_blank",rel:"noreferrer"},"For more information, see PTX ISA")],-1)),t[131]||(t[131]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L2917-L2924",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",Z,[e("summary",null,[t[132]||(t[132]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.tcgen05_alloc-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.nvvm.tcgen05_alloc-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.tcgen05_alloc")],-1)),t[133]||(t[133]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[134]||(t[134]=l("",3))]),e("details",O,[e("summary",null,[t[135]||(t[135]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.tcgen05_dealloc-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.nvvm.tcgen05_dealloc-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.tcgen05_dealloc")],-1)),t[136]||(t[136]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[137]||(t[137]=e("p",null,[e("code",null,"tcgen05_dealloc")],-1)),t[138]||(t[138]=e("p",null,[a("The "),e("code",null,"tcgen05.dealloc"),a(" Op de-allocates the tensor core memory specified by "),e("code",null,"tmemAddr"),a(", which must be from a previous tensor memory allocation. The "),e("code",null,"nCols"),a(" operand specifies the number of columns to be de-allocated, and it must be a power-of-two. "),e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions",target:"_blank",rel:"noreferrer"},"For more information, refer to the PTX ISA")],-1)),t[139]||(t[139]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L2993-L3002",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",E,[e("summary",null,[t[140]||(t[140]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.tcgen05_relinquish_alloc_permit-Tuple{}",href:"#Reactant.MLIR.Dialects.nvvm.tcgen05_relinquish_alloc_permit-Tuple{}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.tcgen05_relinquish_alloc_permit")],-1)),t[141]||(t[141]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[142]||(t[142]=e("p",null,[e("code",null,"tcgen05_relinquish_alloc_permit")],-1)),t[143]||(t[143]=e("p",null,[a("The "),e("code",null,"tcgen05.relinquish_alloc_permit"),a(" Op specifies that the CTA of the executing thread is relinquishing the right to allocate Tensor Memory. So, it is illegal for a CTA to perform "),e("code",null,"tcgen05.alloc"),a(" after any of its constituent threads execute "),e("code",null,"tcgen05.relinquish_alloc_permit"),a(". "),e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions",target:"_blank",rel:"noreferrer"},"For more information, refer to the PTX ISA")],-1)),t[144]||(t[144]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L3023-L3032",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",z,[e("summary",null,[t[145]||(t[145]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.wgmma_commit_group_sync_aligned-Tuple{}",href:"#Reactant.MLIR.Dialects.nvvm.wgmma_commit_group_sync_aligned-Tuple{}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.wgmma_commit_group_sync_aligned")],-1)),t[146]||(t[146]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[147]||(t[147]=e("p",null,[e("code",null,"wgmma_commit_group_sync_aligned")],-1)),t[148]||(t[148]=e("p",null,"Commits all prior uncommitted warpgroup level matrix multiplication operations.",-1)),t[149]||(t[149]=e("p",null,[e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group",target:"_blank",rel:"noreferrer"},"For more information, see PTX ISA")],-1)),t[150]||(t[150]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L3329-L3335",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",X,[e("summary",null,[t[151]||(t[151]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.wgmma_fence_aligned-Tuple{}",href:"#Reactant.MLIR.Dialects.nvvm.wgmma_fence_aligned-Tuple{}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.wgmma_fence_aligned")],-1)),t[152]||(t[152]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[153]||(t[153]=e("p",null,[e("code",null,"wgmma_fence_aligned")],-1)),t[154]||(t[154]=e("p",null,"Enforce an ordering of register accesses between warpgroup level matrix multiplication and other operations.",-1)),t[155]||(t[155]=e("p",null,[e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence",target:"_blank",rel:"noreferrer"},"For more information, see PTX ISA")],-1)),t[156]||(t[156]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L3302-L3309",target:"_blank",rel:"noreferrer"},"source")],-1))]),e("details",q,[e("summary",null,[t[157]||(t[157]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.wgmma_mma_async-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.nvvm.wgmma_mma_async-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.wgmma_mma_async")],-1)),t[158]||(t[158]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[159]||(t[159]=l("",6))]),e("details",B,[e("summary",null,[t[160]||(t[160]=e("a",{id:"Reactant.MLIR.Dialects.nvvm.wgmma_wait_group_sync_aligned-Tuple{}",href:"#Reactant.MLIR.Dialects.nvvm.wgmma_wait_group_sync_aligned-Tuple{}"},[e("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.nvvm.wgmma_wait_group_sync_aligned")],-1)),t[161]||(t[161]=a()),s(n,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),t[162]||(t[162]=e("p",null,[e("code",null,"wgmma_wait_group_sync_aligned")],-1)),t[163]||(t[163]=e("p",null,"Signal the completion of a preceding warpgroup operation.",-1)),t[164]||(t[164]=e("p",null,[e("a",{href:"https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group",target:"_blank",rel:"noreferrer"},"For more information, see PTX ISA")],-1)),t[165]||(t[165]=e("p",null,[e("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/4ece29d643d04456782b8d03d36b925e0c7cd55c/src/mlir/Dialects/Nvvm.jl#L3460-L3466",target:"_blank",rel:"noreferrer"},"source")],-1))])])}const te=i(c,[["render",$]]);export{ee as __pageData,te as default};
