import{_ as r,C as i,c,o as d,j as t,a as l,a2 as s,G as o,w as n}from"./chunks/framework.B7EsJdJP.js";const O=JSON.parse('{"title":"MPI Dialect","description":"","frontmatter":{},"headers":[],"relativePath":"api/dialects/mpi.md","filePath":"api/dialects/mpi.md","lastUpdated":null}'),u={name:"api/dialects/mpi.md"},p={class:"jldocstring custom-block"},m={class:"jldocstring custom-block"},R={class:"jldocstring custom-block"},f={class:"jldocstring custom-block"},I={class:"jldocstring custom-block"},M={class:"jldocstring custom-block"},b={class:"jldocstring custom-block"},L={class:"jldocstring custom-block"},y={class:"jldocstring custom-block"},T={class:"jldocstring custom-block"},j={class:"jldocstring custom-block"},D={class:"jldocstring custom-block"},k={class:"jldocstring custom-block"};function g(P,e,h,v,V,_){const a=i("Badge");return d(),c("div",null,[e[69]||(e[69]=t("h1",{id:"MPI-Dialect",tabindex:"-1"},[l("MPI Dialect "),t("a",{class:"header-anchor",href:"#MPI-Dialect","aria-label":'Permalink to "MPI Dialect {#MPI-Dialect}"'},"​")],-1)),e[70]||(e[70]=t("p",null,[l("Refer to the "),t("a",{href:"https://mlir.llvm.org/docs/Dialects/MPI/",target:"_blank",rel:"noreferrer"},"official documentation"),l(" for more details.")],-1)),t("details",p,[t("summary",null,[e[0]||(e[0]=t("a",{id:"Reactant.MLIR.Dialects.mpi.allreduce-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.allreduce-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.allreduce")],-1)),e[1]||(e[1]=l()),o(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[3]||(e[3]=s("<p><code>allreduce</code></p><p>MPI_Allreduce performs a reduction operation on the values in the sendbuf array and stores the result in the recvbuf array. The operation is performed across all processes in the communicator.</p><p>The <code>op</code> attribute specifies the reduction operation to be performed. Currently only the <code>MPI_Op</code> predefined in the standard (e.g. <code>MPI_SUM</code>) are supported.</p><p>Communicators other than <code>MPI_COMM_WORLD</code> are not supported for now.</p><p>This operation can optionally return an <code>!mpi.retval</code> value that can be used to check for errors.</p>",5)),o(a,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[2]||(e[2]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/0cfcef7182ec2cc6021e95cec7238bf447b810d2/src/mlir/Dialects/MPI.jl#L16-L31",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",m,[t("summary",null,[e[4]||(e[4]=t("a",{id:"Reactant.MLIR.Dialects.mpi.barrier-Tuple{}",href:"#Reactant.MLIR.Dialects.mpi.barrier-Tuple{}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.barrier")],-1)),e[5]||(e[5]=l()),o(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[7]||(e[7]=t("p",null,[t("code",null,"barrier")],-1)),e[8]||(e[8]=t("p",null,"MPI_Barrier blocks execution until all processes in the communicator have reached this routine.",-1)),e[9]||(e[9]=t("p",null,[l("Communicators other than "),t("code",null,"MPI_COMM_WORLD"),l(" are not supported for now.")],-1)),e[10]||(e[10]=t("p",null,[l("This operation can optionally return an "),t("code",null,"!mpi.retval"),l(" value that can be used to check for errors.")],-1)),o(a,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[6]||(e[6]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/0cfcef7182ec2cc6021e95cec7238bf447b810d2/src/mlir/Dialects/MPI.jl#L58-L68",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",R,[t("summary",null,[e[11]||(e[11]=t("a",{id:"Reactant.MLIR.Dialects.mpi.comm_rank-Tuple{}",href:"#Reactant.MLIR.Dialects.mpi.comm_rank-Tuple{}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.comm_rank")],-1)),e[12]||(e[12]=l()),o(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[14]||(e[14]=t("p",null,[t("code",null,"comm_rank")],-1)),e[15]||(e[15]=t("p",null,[l("Communicators other than "),t("code",null,"MPI_COMM_WORLD"),l(" are not supported for now.")],-1)),e[16]||(e[16]=t("p",null,[l("This operation can optionally return an "),t("code",null,"!mpi.retval"),l(" value that can be used to check for errors.")],-1)),o(a,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[13]||(e[13]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/0cfcef7182ec2cc6021e95cec7238bf447b810d2/src/mlir/Dialects/MPI.jl#L89-L96",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",f,[t("summary",null,[e[17]||(e[17]=t("a",{id:"Reactant.MLIR.Dialects.mpi.comm_size-Tuple{}",href:"#Reactant.MLIR.Dialects.mpi.comm_size-Tuple{}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.comm_size")],-1)),e[18]||(e[18]=l()),o(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[20]||(e[20]=t("p",null,[t("code",null,"comm_size")],-1)),e[21]||(e[21]=t("p",null,[l("Communicators other than "),t("code",null,"MPI_COMM_WORLD"),l(" are not supported for now.")],-1)),e[22]||(e[22]=t("p",null,[l("This operation can optionally return an "),t("code",null,"!mpi.retval"),l(" value that can be used to check for errors.")],-1)),o(a,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[19]||(e[19]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/0cfcef7182ec2cc6021e95cec7238bf447b810d2/src/mlir/Dialects/MPI.jl#L119-L126",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",I,[t("summary",null,[e[23]||(e[23]=t("a",{id:"Reactant.MLIR.Dialects.mpi.error_class-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.error_class-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.error_class")],-1)),e[24]||(e[24]=l()),o(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[26]||(e[26]=t("p",null,[t("code",null,"error_class")],-1)),e[27]||(e[27]=t("p",null,[t("code",null,"MPI_Error_class"),l(" maps return values from MPI calls to a set of well-known MPI error classes.")],-1)),o(a,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[25]||(e[25]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/0cfcef7182ec2cc6021e95cec7238bf447b810d2/src/mlir/Dialects/MPI.jl#L149-L154",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",M,[t("summary",null,[e[28]||(e[28]=t("a",{id:"Reactant.MLIR.Dialects.mpi.finalize-Tuple{}",href:"#Reactant.MLIR.Dialects.mpi.finalize-Tuple{}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.finalize")],-1)),e[29]||(e[29]=l()),o(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[31]||(e[31]=t("p",null,[t("code",null,"finalize")],-1)),e[32]||(e[32]=t("p",null,"This function cleans up the MPI state. Afterwards, no MPI methods may be invoked (excpet for MPI_Get_version, MPI_Initialized, and MPI_Finalized). Notably, MPI_Init cannot be called again in the same program.",-1)),e[33]||(e[33]=t("p",null,[l("This operation can optionally return an "),t("code",null,"!mpi.retval"),l(" value that can be used to check for errors.")],-1)),o(a,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[30]||(e[30]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/0cfcef7182ec2cc6021e95cec7238bf447b810d2/src/mlir/Dialects/MPI.jl#L174-L183",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",b,[t("summary",null,[e[34]||(e[34]=t("a",{id:"Reactant.MLIR.Dialects.mpi.init-Tuple{}",href:"#Reactant.MLIR.Dialects.mpi.init-Tuple{}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.init")],-1)),e[35]||(e[35]=l()),o(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[37]||(e[37]=t("p",null,[t("code",null,"init")],-1)),e[38]||(e[38]=t("p",null,"This operation must preceed most MPI calls (except for very few exceptions, please consult with the MPI specification on these).",-1)),e[39]||(e[39]=t("p",null,"Passing &argc, &argv is not supported currently.",-1)),e[40]||(e[40]=t("p",null,[l("This operation can optionally return an "),t("code",null,"!mpi.retval"),l(" value that can be used to check for errors.")],-1)),o(a,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[36]||(e[36]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/0cfcef7182ec2cc6021e95cec7238bf447b810d2/src/mlir/Dialects/MPI.jl#L284-L294",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",L,[t("summary",null,[e[41]||(e[41]=t("a",{id:"Reactant.MLIR.Dialects.mpi.irecv-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.irecv-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.irecv")],-1)),e[42]||(e[42]=l()),o(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[44]||(e[44]=s("<p><code>irecv</code></p><p>MPI_Irecv begins a non-blocking receive of <code>size</code> elements of type <code>dtype</code> from rank <code>dest</code>. The <code>tag</code> value and communicator enables the library to determine the matching of multiple sends and receives between the same ranks.</p><p>Communicators other than <code>MPI_COMM_WORLD</code> are not supported for now.</p><p>This operation can optionally return an <code>!mpi.retval</code> value that can be used to check for errors.</p>",4)),o(a,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[43]||(e[43]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/0cfcef7182ec2cc6021e95cec7238bf447b810d2/src/mlir/Dialects/MPI.jl#L204-L216",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",y,[t("summary",null,[e[45]||(e[45]=t("a",{id:"Reactant.MLIR.Dialects.mpi.isend-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.isend-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.isend")],-1)),e[46]||(e[46]=l()),o(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[48]||(e[48]=s("<p><code>isend</code></p><p>MPI_Isend begins a non-blocking send of <code>size</code> elements of type <code>dtype</code> to rank <code>dest</code>. The <code>tag</code> value and communicator enables the library to determine the matching of multiple sends and receives between the same ranks.</p><p>Communicators other than <code>MPI_COMM_WORLD</code> are not supported for now.</p><p>This operation can optionally return an <code>!mpi.retval</code> value that can be used to check for errors.</p>",4)),o(a,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[47]||(e[47]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/0cfcef7182ec2cc6021e95cec7238bf447b810d2/src/mlir/Dialects/MPI.jl#L244-L256",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",T,[t("summary",null,[e[49]||(e[49]=t("a",{id:"Reactant.MLIR.Dialects.mpi.recv-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.recv-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.recv")],-1)),e[50]||(e[50]=l()),o(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[52]||(e[52]=s("<p><code>recv</code></p><p>MPI_Recv performs a blocking receive of <code>size</code> elements of type <code>dtype</code> from rank <code>source</code>. The <code>tag</code> value and communicator enables the library to determine the matching of multiple sends and receives between the same ranks.</p><p>Communicators other than <code>MPI_COMM_WORLD</code> are not supported for now. The MPI_Status is set to <code>MPI_STATUS_IGNORE</code>, as the status object is not yet ported to MLIR.</p><p>This operation can optionally return an <code>!mpi.retval</code> value that can be used to check for errors.</p>",4)),o(a,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[51]||(e[51]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/0cfcef7182ec2cc6021e95cec7238bf447b810d2/src/mlir/Dialects/MPI.jl#L315-L329",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",j,[t("summary",null,[e[53]||(e[53]=t("a",{id:"Reactant.MLIR.Dialects.mpi.retval_check-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.retval_check-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.retval_check")],-1)),e[54]||(e[54]=l()),o(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[56]||(e[56]=t("p",null,[t("code",null,"retval_check")],-1)),e[57]||(e[57]=t("p",null,[l("This operation compares MPI status codes to known error class constants such as "),t("code",null,"MPI_SUCCESS"),l(", or "),t("code",null,"MPI_ERR_COMM"),l(".")],-1)),o(a,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[55]||(e[55]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/0cfcef7182ec2cc6021e95cec7238bf447b810d2/src/mlir/Dialects/MPI.jl#L356-L361",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",D,[t("summary",null,[e[58]||(e[58]=t("a",{id:"Reactant.MLIR.Dialects.mpi.send-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.send-Tuple{Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value, Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.send")],-1)),e[59]||(e[59]=l()),o(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[61]||(e[61]=s("<p><code>send</code></p><p>MPI_Send performs a blocking send of <code>size</code> elements of type <code>dtype</code> to rank <code>dest</code>. The <code>tag</code> value and communicator enables the library to determine the matching of multiple sends and receives between the same ranks.</p><p>Communicators other than <code>MPI_COMM_WORLD</code> are not supported for now.</p><p>This operation can optionally return an <code>!mpi.retval</code> value that can be used to check for errors.</p>",4)),o(a,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[60]||(e[60]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/0cfcef7182ec2cc6021e95cec7238bf447b810d2/src/mlir/Dialects/MPI.jl#L381-L392",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})]),t("details",k,[t("summary",null,[e[62]||(e[62]=t("a",{id:"Reactant.MLIR.Dialects.mpi.wait-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.mpi.wait-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.mpi.wait")],-1)),e[63]||(e[63]=l()),o(a,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[65]||(e[65]=t("p",null,[t("code",null,"wait")],-1)),e[66]||(e[66]=t("p",null,"MPI_Wait blocks execution until the request has completed.",-1)),e[67]||(e[67]=t("p",null,[l("The MPI_Status is set to "),t("code",null,"MPI_STATUS_IGNORE"),l(", as the status object is not yet ported to MLIR.")],-1)),e[68]||(e[68]=t("p",null,[l("This operation can optionally return an "),t("code",null,"!mpi.retval"),l(" value that can be used to check for errors.")],-1)),o(a,{type:"info",class:"source-link",text:"source"},{default:n(()=>e[64]||(e[64]=[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/0cfcef7182ec2cc6021e95cec7238bf447b810d2/src/mlir/Dialects/MPI.jl#L419-L429",target:"_blank",rel:"noreferrer"},"source",-1)])),_:1})])])}const C=r(u,[["render",g]]);export{O as __pageData,C as default};
