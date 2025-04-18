import{_ as i,C as l,c as r,o as d,j as t,a,a2 as n,G as o}from"./chunks/framework.Og7vQWlB.js";const v=JSON.parse('{"title":"Shardy Dialect","description":"","frontmatter":{},"headers":[],"relativePath":"api/shardy.md","filePath":"api/shardy.md","lastUpdated":null}'),p={name:"api/shardy.md"},h={class:"jldocstring custom-block"},c={class:"jldocstring custom-block"},u={class:"jldocstring custom-block"},g={class:"jldocstring custom-block"},f={class:"jldocstring custom-block"},m={class:"jldocstring custom-block"},b={class:"jldocstring custom-block"},y={class:"jldocstring custom-block"},_={class:"jldocstring custom-block"},R={class:"jldocstring custom-block"},T={class:"jldocstring custom-block"};function I(x,e,D,j,L,w){const s=l("Badge");return d(),r("div",null,[e[42]||(e[42]=t("h1",{id:"Shardy-Dialect",tabindex:"-1"},[a("Shardy Dialect "),t("a",{class:"header-anchor",href:"#Shardy-Dialect","aria-label":'Permalink to "Shardy Dialect {#Shardy-Dialect}"'},"​")],-1)),e[43]||(e[43]=t("p",null,[a("Refer to the "),t("a",{href:"https://openxla.org/shardy",target:"_blank",rel:"noreferrer"},"official documentation"),a(" for more details.")],-1)),t("details",h,[t("summary",null,[e[0]||(e[0]=t("a",{id:"Reactant.MLIR.Dialects.sdy.all_gather-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.sdy.all_gather-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.sdy.all_gather")],-1)),e[1]||(e[1]=a()),o(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[2]||(e[2]=n(`<p><code>all_gather</code></p><p>Gathers chunks of a tensor along axes specified in <code>gathering_axes</code>.</p><p>The <code>gathering_axes</code> is a list of lists of axes. The outer list is over the dimensions of the tensor. Each inner list specifies the axes along which a separate gather should be performed on the respective dimension. It will be applied to the sharding of the operand (<code>tensor</code>) to obtain the sharding of the result (<code>out_sharding</code>).</p><p>Note that <code>out_sharding</code> is not used to determine the sharding of the result. Instead, the sharding of the result is determined by the sharding of the operand and the <code>gathering_axes</code>, and <code>out_sharding</code> must match this inferred sharding.</p><p><strong>Example</strong></p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>%1 = stablehlo.tanh(%0) {sdy.sharding = #sdy.sharding_per_value&lt;[&lt;@mesh, [{&quot;a&quot;, &quot;b&quot;, &quot;c&quot;}, {}, {&quot;d&quot;}\\]&gt;]&gt;} : tensor&lt;8x8xf32&gt;</span></span>
<span class="line"><span>%2 = sdy.all_gather [{&quot;b&quot;, &quot;c&quot;}, {}, {&quot;d&quot;}\\] %1 to_sharding=&lt;@mesh, [{&quot;a&quot;}, {}, {}\\]&gt; : tensor&lt;8x8xf32&gt;</span></span></code></pre></div><p><strong>Constraints:</strong></p><ul><li><p>Elements in <code>gathering_axes</code> must satisfy the constraints listed in <code>AxisRefListAttr</code>.</p></li><li><p><code>out_sharding</code> must satisfy the constraints listed in <code>TensorShardingAttr</code>.</p></li><li><p>The operand must have a sharding.</p></li><li><p>Both operand and result shardings should be bound to the same <code>MeshAttr</code>.</p></li><li><p>Applying <code>gathering_axes</code> to the operand sharding gets <code>out_sharding</code>.</p></li></ul><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/9339756f1ea494409f8163350c91237db9ff080a/src/mlir/Dialects/Shardy.jl#L16-L46" target="_blank" rel="noreferrer">source</a></p>`,9))]),t("details",c,[t("summary",null,[e[3]||(e[3]=t("a",{id:"Reactant.MLIR.Dialects.sdy.all_slice-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.sdy.all_slice-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.sdy.all_slice")],-1)),e[4]||(e[4]=a()),o(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[5]||(e[5]=n(`<p><code>all_slice</code></p><p>Slices chunks of a tensor along axes specified in <code>slicing_axes</code>. There is an algebric duality between <code>sdy.all_slice</code> and <code>sdy.all_gather</code>.</p><p>The <code>slicing_axes</code> is a list of lists of axes. The outer list is over the dimensions of the tensor. Each inner list specifies the axes along which a slice should be performed on the respective dimension. It will be applied to the sharding of the operand (<code>tensor</code>) to obtain the sharding of the result (<code>out_sharding</code>).</p><p>Note that <code>out_sharding</code> is not used to determine the sharding of the result. Instead, the sharding of the result is determined by the sharding of the operand and the <code>slicing_axes</code>, and <code>out_sharding</code> must match this inferred sharding.</p><p><strong>Example</strong></p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>%1 = stablehlo.tanh(%0) {sdy.sharding = #sdy.sharding_per_value&lt;[&lt;@mesh, [{&quot;a&quot;}, {}, {}\\]&gt;]&gt;} : tensor&lt;8x8xf32&gt;</span></span>
<span class="line"><span>%2 = sdy.all_slice [{&quot;b&quot;, &quot;c&quot;}, {}, {&quot;d&quot;}\\] %1 to_sharding=&lt;@mesh, [{&quot;a&quot;, &quot;b&quot;, &quot;c&quot;}, {}, {&quot;d&quot;}\\]&gt; : tensor&lt;8x8xf32&gt;</span></span></code></pre></div><p><strong>Constraints:</strong></p><ul><li><p>Elements in <code>slicing_axes</code> must satisfy the constraints listed in <code>AxisRefListAttr</code>.</p></li><li><p><code>out_sharding</code> must satisfy the constraints listed in <code>TensorShardingAttr</code>.</p></li><li><p>The operand must have a sharding.</p></li><li><p>Both operand and result shardings should be bound to the same <code>MeshAttr</code>.</p></li><li><p>Applying <code>slicing_axes</code> to the operand sharding gets <code>out_sharding</code>.</p></li></ul><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/9339756f1ea494409f8163350c91237db9ff080a/src/mlir/Dialects/Shardy.jl#L76-L107" target="_blank" rel="noreferrer">source</a></p>`,9))]),t("details",u,[t("summary",null,[e[6]||(e[6]=t("a",{id:"Reactant.MLIR.Dialects.sdy.constant-Tuple{}",href:"#Reactant.MLIR.Dialects.sdy.constant-Tuple{}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.sdy.constant")],-1)),e[7]||(e[7]=a()),o(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[8]||(e[8]=n('<p><code>constant</code></p><p>Produces an <code>output</code> tensor from a constant <code>value</code>.</p><p>See: <a href="https://github.com/openxla/stablehlo/blob/main/docs/spec.md#constant" target="_blank" rel="noreferrer">https://github.com/openxla/stablehlo/blob/main/docs/spec.md#constant</a></p><p>NOTE: SDY defines its own constant op that isn&#39;t ConstantLike and doesn&#39;t have a folder, so that we&#39;ll be able to duplicate constants without any greedy pattern rewriter folding them back into a single constant. In this way, constants can be sharded differently for every use, and no propagation is done between constants (or constant expressions).</p><p><strong>Example</strong></p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>%output = sdy.constant dense&lt;[[0.0, 1.0], [2.0, 3.0]]&gt; : tensor&lt;2x2xf32&gt;</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/9339756f1ea494409f8163350c91237db9ff080a/src/mlir/Dialects/Shardy.jl#L137-L155" target="_blank" rel="noreferrer">source</a></p>',7))]),t("details",g,[t("summary",null,[e[9]||(e[9]=t("a",{id:"Reactant.MLIR.Dialects.sdy.data_flow_edge-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.sdy.data_flow_edge-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.sdy.data_flow_edge")],-1)),e[10]||(e[10]=a()),o(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[11]||(e[11]=n(`<p><code>data_flow_edge</code></p><p>A data flow edge of some op X defines a bridge between a set of sources (each is either an operand of X or an operand of X&#39;s block terminator) and a set of targets (each is either a result of X or a block argument of X), such that all sources and targets should be sharded in the same way.</p><p>An op can have multiple data flow edges that are orthogonal to one another.</p><p>For example:</p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>  y_0, ..., y_n = while (x_0, ..., x_n)</span></span>
<span class="line"><span>                  ((pred_arg_0,... , pred_arg_n) { ... })</span></span>
<span class="line"><span>                  ((body_arg_0,..., body_arg_n) {</span></span>
<span class="line"><span>                    ...</span></span>
<span class="line"><span>                    return return_value_0, ..., return_value_n</span></span>
<span class="line"><span>                  })</span></span></code></pre></div><p>This while op has n data flow edges, the i-th data flow edges is between sources <code>x_i</code>, <code>return_value_i</code> and targets <code>y_i</code>, <code>pred_arg_i</code>, <code>body_arg_i</code>.</p><p>An <code>sdy.data_flow_edge</code> takes as input the owner of an edge (can be any of the targets, but preferably an op result rather than a block argument), which shouldn&#39;t have any other uses. This op isn&#39;t pure because it can take an input that originally didn&#39;t have any uses.</p><p>The <code>sdy.data_flow_edge</code> also holds an optional sharding for all targets of the edge, and that sharding should be updated instead of the targets&#39; sharding (if can be attached) during propagation. This is useful when an op has many edges, as it&#39;s much more efficient to:</p><ul><li><p>propagate through each edge separately.</p></li><li><p>update the sharding of each edge separately instead of all targets at once (e.g. an op has a single immutable <code>TensorShardingPerValueAttr</code> for result shardings).</p></li><li><p>add each edge to the worklist separately when the sharding of a source has changed.</p></li></ul><p>Propagation will propagate shardings between all sources and targets of a <code>sdy.data_flow_edge</code> as if it was a regular op with the sources as operands and targets as results, and an identity <code>sdy.op_sharding_rule</code>. That means that forward propagation is from sources to targets and backwards propagation is from targets to sources.</p><p>We don&#39;t allow the input of a <code>sdy.data_flow_edge</code> to be defined by an <code>SdyDialect</code> op, so we can assume that it&#39;s defined by an op that has unregistered <code>sdy.sharding</code> attribute.</p><p>NOTE: it&#39;s NOT the responsibility of the <code>sdy.data_flow_edge</code> to link between sources and targets, it&#39;s simply attached to the owner of the edge. The op that this edge is bound to (while in the example above) is responsible for providing this information.</p><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/9339756f1ea494409f8163350c91237db9ff080a/src/mlir/Dialects/Shardy.jl#L176-L231" target="_blank" rel="noreferrer">source</a></p>`,13))]),t("details",f,[t("summary",null,[e[12]||(e[12]=t("a",{id:"Reactant.MLIR.Dialects.sdy.manual_computation-Tuple{Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.sdy.manual_computation-Tuple{Vector{Reactant.MLIR.IR.Value}}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.sdy.manual_computation")],-1)),e[13]||(e[13]=a()),o(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[14]||(e[14]=n('<p><code>manual_computation</code></p><p>Jump into a region written in terms of per-device local code with explicit collectives, where logical shapes match local per-device physical buffer shapes and collectives correspond exactly to physical cross-device communication.</p><p>The body is local wrt the manual_axes. Propagation will occur through the body on any free axes - those not in the manual_axes list.</p><p><strong>Constraints:</strong></p><ul><li><p>Elements in <code>in_shardings</code> and <code>out_shardings</code> must satisfy the constraints listed in <code>TensorShardingAttr</code>.</p></li><li><p>The number of global and local tensor inputs/outputs of the op region must match.</p></li><li><p>The manual axes must come before any free axes in each dim sharding.</p></li><li><p>The global and local shapes of the op regions arguments/results must match.</p></li><li><p>No manual axes are split.</p></li></ul><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/9339756f1ea494409f8163350c91237db9ff080a/src/mlir/Dialects/Shardy.jl#L258-L275" target="_blank" rel="noreferrer">source</a></p>',6))]),t("details",m,[t("summary",null,[e[15]||(e[15]=t("a",{id:"Reactant.MLIR.Dialects.sdy.mesh-Tuple{}",href:"#Reactant.MLIR.Dialects.sdy.mesh-Tuple{}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.sdy.mesh")],-1)),e[16]||(e[16]=a()),o(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[17]||(e[17]=t("p",null,[t("code",null,"mesh")],-1)),e[18]||(e[18]=t("p",null,[a("Defines a new named mesh. All meshes in a module must have the same number of devices (except for meshes with a single device_id). The mesh is a "),t("code",null,"Symbol"),a(" operation that appears in the module's "),t("code",null,"SymbolTable"),a(" and can be referenced by its "),t("code",null,"name"),a(".")],-1)),e[19]||(e[19]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/9339756f1ea494409f8163350c91237db9ff080a/src/mlir/Dialects/Shardy.jl#L307-L314",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",b,[t("summary",null,[e[20]||(e[20]=t("a",{id:"Reactant.MLIR.Dialects.sdy.named_computation-Tuple{Vector{Reactant.MLIR.IR.Value}}",href:"#Reactant.MLIR.Dialects.sdy.named_computation-Tuple{Vector{Reactant.MLIR.IR.Value}}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.sdy.named_computation")],-1)),e[21]||(e[21]=a()),o(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[22]||(e[22]=n(`<p><code>named_computation</code></p><p>Groups a computation, i.e. a block of operations, and gives it a name. Propagation will flow in/out of the region as if everything was inlined.</p><p>This can be used to handle propagating through call instructions to other functions. Any users of Shardy should write an import/export pass that converts their call ops to <code>sdy.named_computation</code> ops, duplicating/copying the body of the called function into the body of the <code>named_computation</code>.</p><p>The type of each block arguments and returned values in the region must be the same as the type of the operands and results type of the op.</p><p><strong>Example</strong></p><div class="language-mlir vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">mlir</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>%1 = sdy.named_computation&lt;&quot;foo&quot;&gt;(%0) (%arg1: tensor&lt;16x32xf32&gt;) {</span></span>
<span class="line"><span>  sdy.return %arg1 : tensor&lt;16x32xf32&gt;</span></span>
<span class="line"><span>} : (tensor&lt;16x32xf32&gt;) -&gt; tensor&lt;16x32xf32&gt;</span></span></code></pre></div><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/9339756f1ea494409f8163350c91237db9ff080a/src/mlir/Dialects/Shardy.jl#L336-L357" target="_blank" rel="noreferrer">source</a></p>`,7))]),t("details",y,[t("summary",null,[e[23]||(e[23]=t("a",{id:"Reactant.MLIR.Dialects.sdy.propagation_barrier-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.sdy.propagation_barrier-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.sdy.propagation_barrier")],-1)),e[24]||(e[24]=a()),o(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[25]||(e[25]=n('<p><code>propagation_barrier</code></p><p>This op operates like an identity op, outputting the same value it took as input. But in terms of propagation, this will only allow propagation to flow through it in a certain direction.</p><p>This prevents shardings from being propagated between the uses of the result of the barrier op and its operand.</p><ul><li><p><code>FORWARD</code> means shardings can only flow from the operand to the result.</p></li><li><p><code>BACKWARD</code> means shardings can only flow from the result to the operand.</p></li><li><p><code>NONE</code> means no sharding can propagate through this op.</p></li><li><p>Cannot specify <code>BOTH</code>, as this op would be redundant.</p></li></ul><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/9339756f1ea494409f8163350c91237db9ff080a/src/mlir/Dialects/Shardy.jl#L389-L403" target="_blank" rel="noreferrer">source</a></p>',5))]),t("details",_,[t("summary",null,[e[26]||(e[26]=t("a",{id:"Reactant.MLIR.Dialects.sdy.reshard-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.sdy.reshard-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.sdy.reshard")],-1)),e[27]||(e[27]=a()),o(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[28]||(e[28]=n('<p><code>reshard</code></p><p>Reshards the input tensor with the specified sharding, which is different from the input tensor&#39;s existing sharding.</p><p>Both ShardingConstraintOp and ReshardOp attach a sharding to a tensor. Their lifespan is:</p><ol><li><p>Before sharding propagation, ShardingConstraintOp is added by users.</p></li><li><p>Sharding propagation consumes ShardingConstraintOp. There is no ShardingConstraintOp in the results of sharding propagation. Instead, ReshardOp may be added if needed.</p></li><li><p>A partitioner converts a ReshardOp into a collective op (or an identity op). There should be no ReshardOp in the results of the partitioner.</p></li></ol><p>// TODO(b/331680067). Add a canonicalization pattern to remove redundant // reshard ops.</p><p><a href="https://github.com/EnzymeAD/Reactant.jl/blob/9339756f1ea494409f8163350c91237db9ff080a/src/mlir/Dialects/Shardy.jl#L429-L446" target="_blank" rel="noreferrer">source</a></p>',6))]),t("details",R,[t("summary",null,[e[29]||(e[29]=t("a",{id:"Reactant.MLIR.Dialects.sdy.sharding_constraint-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.sdy.sharding_constraint-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.sdy.sharding_constraint")],-1)),e[30]||(e[30]=a()),o(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[31]||(e[31]=t("p",null,[t("code",null,"sharding_constraint")],-1)),e[32]||(e[32]=t("p",null,"Attaches a sharding to an intermediate tensor (e.g. the result of a matmul) to indicate that this is how that tensor, or a subset of its uses, should be sharded.",-1)),e[33]||(e[33]=t("p",null,"If the sharding has open dimensions and unconstraint axes, it means the tensor can be further sharded along the open dimensions.",-1)),e[34]||(e[34]=t("p",null,"This op can either:",-1)),e[35]||(e[35]=t("ul",null,[t("li",null,[t("p",null,"Have no uses (dangling) - which means the attached sharding is how the input tensor itself should be sharded.")]),t("li",null,[t("p",null,"Have uses - which means the attached sharding is how the uses of the sharding constraint op should be sharded, while other uses of the input tensor might have a different sharding (if the input tensor has no other uses then the behavior is the same as the no uses case).")])],-1)),e[36]||(e[36]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/9339756f1ea494409f8163350c91237db9ff080a/src/mlir/Dialects/Shardy.jl#L488-L505",target:"_blank",rel:"noreferrer"},"source")],-1))]),t("details",T,[t("summary",null,[e[37]||(e[37]=t("a",{id:"Reactant.MLIR.Dialects.sdy.sharding_group-Tuple{Reactant.MLIR.IR.Value}",href:"#Reactant.MLIR.Dialects.sdy.sharding_group-Tuple{Reactant.MLIR.IR.Value}"},[t("span",{class:"jlbinding"},"Reactant.MLIR.Dialects.sdy.sharding_group")],-1)),e[38]||(e[38]=a()),o(s,{type:"info",class:"jlObjectType jlMethod",text:"Method"})]),e[39]||(e[39]=t("p",null,[t("code",null,"sharding_group")],-1)),e[40]||(e[40]=t("p",null,"This op provides an interface to assign tensors to sharding groups ( groups of tensors that will be enforced to have identical shardings). During propagation, as soon as one group element is sharded, all other members will be sharded in exactly the same way. This operation takes the argument group ID and returns no result, but instead modifies the internal sharding group representation to add the input tensor to the group with the given ID.",-1)),e[41]||(e[41]=t("p",null,[t("a",{href:"https://github.com/EnzymeAD/Reactant.jl/blob/9339756f1ea494409f8163350c91237db9ff080a/src/mlir/Dialects/Shardy.jl#L528-L538",target:"_blank",rel:"noreferrer"},"source")],-1))])])}const k=i(p,[["render",I]]);export{v as __pageData,k as default};
