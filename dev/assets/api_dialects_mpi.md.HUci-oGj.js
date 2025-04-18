import{_ as r,C as i,c,o as d,j as t,a as l,al as n,G as s,w as o}from"./chunks/framework.B4QBCUam.js";const _=JSON.parse('{"title":"MPI Dialect","description":"","frontmatter":{},"headers":[],"relativePath":"api/dialects/mpi.md","filePath":"api/dialects/mpi.md","lastUpdated":null}'),u={name:"api/dialects/mpi.md"},p={class:"jldocstring custom-block"},m={class:"jldocstring custom-block"},R={class:"jldocstring custom-block"},f={class:"jldocstring custom-block"},I={class:"jldocstring custom-block"},b={class:"jldocstring custom-block"},M={class:"jldocstring custom-block"},y={class:"jldocstring custom-block"},L={class:"jldocstring custom-block"},j={class:"jldocstring custom-block"},T={class:"jldocstring custom-block"},D={class:"jldocstring custom-block"},k={class:"jldocstring custom-block"},g={class:"jldocstring custom-block"},P={class:"jldocstring custom-block"};function v(h,e,V,x,A,z){const a=i("Badge");return d(),c("div",null,[e[77]||(e[77]=t("h1",{id:"MPI-Dialect",tabindex:"-1"},[l("MPI Dialect "),t("a",{class:"header-anchor",href:"#MPI-Dialect","aria-label":'Permalink to "MPI Dialect {#MPI-Dialect}"'},"​")],-1)),e[78]||(e[78]=t("p",null,[l("Refer to the "),t("a",{href:"https://mlir.llvm.org/docs/Dialects/MPI/",target:"_blank",rel:"noreferrer"},"official documentation"),l(" for more details.")],-1)),t("details",p,[t("summary",null,[e[0]||(e[0]=t("a",{id:"Reactant.MLIR.Dialects.mpi.allreduce-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.allreduce-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.allreduce")],-1)),e[1]||(e[1]=l()),s(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[3]||(e[3]=n("<p><code>allreduce</code></p><p>MPI_Allreduce performs a reduction operation on the values in the sendbuf array and stores the result in the recvbuf array. The operation is performed across all processes in the communicator.</p><p>The <code>op</code> attribute specifies the reduction operation to be performed. Currently only the <code>MPI_Op</code> predefined in the standard (e.g. <code>MPI_SUM</code>) are supported.</p><p>This operation can optionally return an <code>!mpi.retval</code> value that can be used to check for errors.</p>",4)),s(a,{type:"info",class:"source-link",text:"source"},{default:o(()=>e[2]||(e[2]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a6e4c404c93150fe758d6b9e1d184a73f2193d85/src/mlir/Dialects/MPI.jl#L16-L29",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",m,[t("summary",null,[e[4]||(e[4]=t("a",{id:"Reactant.MLIR.Dialects.mpi.barrier-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.barrier-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.barrier")],-1)),e[5]||(e[5]=l()),s(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[7]||(e[7]=t("p",null,[t("code",null,"barrier")],-1)),e[8]||(e[8]=t("p",null,"MPI_Barrier blocks execution until all processes in the communicator have reached this routine.",-1)),e[9]||(e[9]=t("p",null,[l("This operation can optionally return an "),t("code",null,"!mpi.retval"),l(" value that can be used to check for errors.")],-1)),s(a,{type:"info",class:"source-link",text:"source"},{default:o(()=>e[6]||(e[6]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a6e4c404c93150fe758d6b9e1d184a73f2193d85/src/mlir/Dialects/MPI.jl#L57-L65",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",R,[t("summary",null,[e[10]||(e[10]=t("a",{id:"Reactant.MLIR.Dialects.mpi.comm_rank-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.comm_rank-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.comm_rank")],-1)),e[11]||(e[11]=l()),s(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[13]||(e[13]=t("p",null,[t("code",null,"comm_rank")],-1)),e[14]||(e[14]=t("p",null,[l("This operation can optionally return an "),t("code",null,"!mpi.retval"),l(" value that can be used to check for errors.")],-1)),s(a,{type:"info",class:"source-link",text:"source"},{default:o(()=>e[12]||(e[12]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a6e4c404c93150fe758d6b9e1d184a73f2193d85/src/mlir/Dialects/MPI.jl#L86-L91",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",f,[t("summary",null,[e[15]||(e[15]=t("a",{id:"Reactant.MLIR.Dialects.mpi.comm_size-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.comm_size-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.comm_size")],-1)),e[16]||(e[16]=l()),s(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[18]||(e[18]=t("p",null,[t("code",null,"comm_size")],-1)),e[19]||(e[19]=t("p",null,[l("This operation can optionally return an "),t("code",null,"!mpi.retval"),l(" value that can be used to check for errors.")],-1)),s(a,{type:"info",class:"source-link",text:"source"},{default:o(()=>e[17]||(e[17]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a6e4c404c93150fe758d6b9e1d184a73f2193d85/src/mlir/Dialects/MPI.jl#L114-L119",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",I,[t("summary",null,[e[20]||(e[20]=t("a",{id:"Reactant.MLIR.Dialects.mpi.comm_split-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.comm_split-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.comm_split")],-1)),e[21]||(e[21]=l()),s(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[23]||(e[23]=t("p",null,[t("code",null,"comm_split")],-1)),e[24]||(e[24]=t("p",null,"This operation splits the communicator into multiple sub-communicators. The color value determines the group of processes that will be part of the new communicator. The key value determines the rank of the calling process in the new communicator.",-1)),e[25]||(e[25]=t("p",null,[l("This operation can optionally return an "),t("code",null,"!mpi.retval"),l(" value that can be used to check for errors.")],-1)),s(a,{type:"info",class:"source-link",text:"source"},{default:o(()=>e[22]||(e[22]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a6e4c404c93150fe758d6b9e1d184a73f2193d85/src/mlir/Dialects/MPI.jl#L142-L152",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",b,[t("summary",null,[e[26]||(e[26]=t("a",{id:"Reactant.MLIR.Dialects.mpi.comm_world-Tuple{}",href:"#Reactant.MLIR.Dialects.mpi.comm_world-Tuple{}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.comm_world")],-1)),e[27]||(e[27]=l()),s(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[29]||(e[29]=t("p",null,[t("code",null,"comm_world")],-1)),e[30]||(e[30]=t("p",null,"This operation returns the predefined MPI_COMM_WORLD communicator.",-1)),s(a,{type:"info",class:"source-link",text:"source"},{default:o(()=>e[28]||(e[28]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a6e4c404c93150fe758d6b9e1d184a73f2193d85/src/mlir/Dialects/MPI.jl#L180-L184",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",M,[t("summary",null,[e[31]||(e[31]=t("a",{id:"Reactant.MLIR.Dialects.mpi.error_class-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.error_class-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.error_class")],-1)),e[32]||(e[32]=l()),s(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[34]||(e[34]=t("p",null,[t("code",null,"error_class")],-1)),e[35]||(e[35]=t("p",null,[t("code",null,"MPI_Error_class"),l(" maps return values from MPI calls to a set of well-known MPI error classes.")],-1)),s(a,{type:"info",class:"source-link",text:"source"},{default:o(()=>e[33]||(e[33]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a6e4c404c93150fe758d6b9e1d184a73f2193d85/src/mlir/Dialects/MPI.jl#L204-L209",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",y,[t("summary",null,[e[36]||(e[36]=t("a",{id:"Reactant.MLIR.Dialects.mpi.finalize-Tuple{}",href:"#Reactant.MLIR.Dialects.mpi.finalize-Tuple{}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.finalize")],-1)),e[37]||(e[37]=l()),s(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[39]||(e[39]=t("p",null,[t("code",null,"finalize")],-1)),e[40]||(e[40]=t("p",null,"This function cleans up the MPI state. Afterwards, no MPI methods may be invoked (excpet for MPI_Get_version, MPI_Initialized, and MPI_Finalized). Notably, MPI_Init cannot be called again in the same program.",-1)),e[41]||(e[41]=t("p",null,[l("This operation can optionally return an "),t("code",null,"!mpi.retval"),l(" value that can be used to check for errors.")],-1)),s(a,{type:"info",class:"source-link",text:"source"},{default:o(()=>e[38]||(e[38]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a6e4c404c93150fe758d6b9e1d184a73f2193d85/src/mlir/Dialects/MPI.jl#L229-L238",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",L,[t("summary",null,[e[42]||(e[42]=t("a",{id:"Reactant.MLIR.Dialects.mpi.init-Tuple{}",href:"#Reactant.MLIR.Dialects.mpi.init-Tuple{}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.init")],-1)),e[43]||(e[43]=l()),s(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[45]||(e[45]=t("p",null,[t("code",null,"init")],-1)),e[46]||(e[46]=t("p",null,"This operation must preceed most MPI calls (except for very few exceptions, please consult with the MPI specification on these).",-1)),e[47]||(e[47]=t("p",null,"Passing &argc, &argv is not supported currently.",-1)),e[48]||(e[48]=t("p",null,[l("This operation can optionally return an "),t("code",null,"!mpi.retval"),l(" value that can be used to check for errors.")],-1)),s(a,{type:"info",class:"source-link",text:"source"},{default:o(()=>e[44]||(e[44]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a6e4c404c93150fe758d6b9e1d184a73f2193d85/src/mlir/Dialects/MPI.jl#L337-L347",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",j,[t("summary",null,[e[49]||(e[49]=t("a",{id:"Reactant.MLIR.Dialects.mpi.irecv-NTuple{4, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.irecv-NTuple{4, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.irecv")],-1)),e[50]||(e[50]=l()),s(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[52]||(e[52]=n("<p><code>irecv</code></p><p>MPI_Irecv begins a non-blocking receive of <code>size</code> elements of type <code>dtype</code> from rank <code>source</code>. The <code>tag</code> value and communicator enables the library to determine the matching of multiple sends and receives between the same ranks.</p><p>This operation can optionally return an <code>!mpi.retval</code> value that can be used to check for errors.</p>",3)),s(a,{type:"info",class:"source-link",text:"source"},{default:o(()=>e[51]||(e[51]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a6e4c404c93150fe758d6b9e1d184a73f2193d85/src/mlir/Dialects/MPI.jl#L259-L269",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",T,[t("summary",null,[e[53]||(e[53]=t("a",{id:"Reactant.MLIR.Dialects.mpi.isend-NTuple{4, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.isend-NTuple{4, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.isend")],-1)),e[54]||(e[54]=l()),s(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[56]||(e[56]=n("<p><code>isend</code></p><p>MPI_Isend begins a non-blocking send of <code>size</code> elements of type <code>dtype</code> to rank <code>dest</code>. The <code>tag</code> value and communicator enables the library to determine the matching of multiple sends and receives between the same ranks.</p><p>This operation can optionally return an <code>!mpi.retval</code> value that can be used to check for errors.</p>",3)),s(a,{type:"info",class:"source-link",text:"source"},{default:o(()=>e[55]||(e[55]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a6e4c404c93150fe758d6b9e1d184a73f2193d85/src/mlir/Dialects/MPI.jl#L298-L308",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",D,[t("summary",null,[e[57]||(e[57]=t("a",{id:"Reactant.MLIR.Dialects.mpi.recv-NTuple{4, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.recv-NTuple{4, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.recv")],-1)),e[58]||(e[58]=l()),s(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[60]||(e[60]=n("<p><code>recv</code></p><p>MPI_Recv performs a blocking receive of <code>size</code> elements of type <code>dtype</code> from rank <code>source</code>. The <code>tag</code> value and communicator enables the library to determine the matching of multiple sends and receives between the same ranks.</p><p>The MPI_Status is set to <code>MPI_STATUS_IGNORE</code>, as the status object is not yet ported to MLIR.</p><p>This operation can optionally return an <code>!mpi.retval</code> value that can be used to check for errors.</p>",4)),s(a,{type:"info",class:"source-link",text:"source"},{default:o(()=>e[59]||(e[59]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a6e4c404c93150fe758d6b9e1d184a73f2193d85/src/mlir/Dialects/MPI.jl#L368-L381",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",k,[t("summary",null,[e[61]||(e[61]=t("a",{id:"Reactant.MLIR.Dialects.mpi.retval_check-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.retval_check-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.retval_check")],-1)),e[62]||(e[62]=l()),s(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[64]||(e[64]=t("p",null,[t("code",null,"retval_check")],-1)),e[65]||(e[65]=t("p",null,[l("This operation compares MPI status codes to known error class constants such as "),t("code",null,"MPI_SUCCESS"),l(", or "),t("code",null,"MPI_ERR_COMM"),l(".")],-1)),s(a,{type:"info",class:"source-link",text:"source"},{default:o(()=>e[63]||(e[63]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a6e4c404c93150fe758d6b9e1d184a73f2193d85/src/mlir/Dialects/MPI.jl#L409-L414",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",g,[t("summary",null,[e[66]||(e[66]=t("a",{id:"Reactant.MLIR.Dialects.mpi.send-NTuple{4, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.send-NTuple{4, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.send")],-1)),e[67]||(e[67]=l()),s(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[69]||(e[69]=n("<p><code>send</code></p><p>MPI_Send performs a blocking send of <code>size</code> elements of type <code>dtype</code> to rank <code>dest</code>. The <code>tag</code> value and communicator enables the library to determine the matching of multiple sends and receives between the same ranks.</p><p>This operation can optionally return an <code>!mpi.retval</code> value that can be used to check for errors.</p>",3)),s(a,{type:"info",class:"source-link",text:"source"},{default:o(()=>e[68]||(e[68]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a6e4c404c93150fe758d6b9e1d184a73f2193d85/src/mlir/Dialects/MPI.jl#L434-L443",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",P,[t("summary",null,[e[70]||(e[70]=t("a",{id:"Reactant.MLIR.Dialects.mpi.wait-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.wait-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.wait")],-1)),e[71]||(e[71]=l()),s(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[73]||(e[73]=t("p",null,[t("code",null,"wait")],-1)),e[74]||(e[74]=t("p",null,"MPI_Wait blocks execution until the request has completed.",-1)),e[75]||(e[75]=t("p",null,[l("The MPI_Status is set to "),t("code",null,"MPI_STATUS_IGNORE"),l(", as the status object is not yet ported to MLIR.")],-1)),e[76]||(e[76]=t("p",null,[l("This operation can optionally return an "),t("code",null,"!mpi.retval"),l(" value that can be used to check for errors.")],-1)),s(a,{type:"info",class:"source-link",text:"source"},{default:o(()=>e[72]||(e[72]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/a6e4c404c93150fe758d6b9e1d184a73f2193d85/src/mlir/Dialects/MPI.jl#L471-L481",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})])])}const E=r(u,[["render",v]]);export{_ as __pageData,E as default};
