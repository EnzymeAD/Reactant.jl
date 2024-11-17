import{_ as s,o as i,c as t,a5 as e}from"./chunks/framework.f6-H1zhv.js";const r=JSON.parse('{"title":"","description":"","frontmatter":{"layout":"home","hero":{"name":"Reactant Docs","text":"TODO","tagline":"TODO","actions":[{"theme":"brand","text":"Tutorials","link":"/tutorials"},{"theme":"alt","text":"API Reference 📚","link":"/api/api"},{"theme":"alt","text":"View on GitHub","link":"https://github.com/EnzymeAD/Reactant.jl"}],"image":{"src":"/logo.svg","alt":"Reactant.jl"}},"features":[{"icon":"🚀","title":"TODO","details":"TODO","link":"TODO"},{"icon":"🧑‍🔬","title":"TODO","details":"TODO","link":"TODO"},{"icon":"🧩","title":"TODO","details":"TODO","link":"TODO"},{"icon":"🧪","title":"TODO","details":"TODO","link":"TODO"}]},"headers":[],"relativePath":"index.md","filePath":"index.md","lastUpdated":null}'),l={name:"index.md"};function n(h,a,p,k,d,o){return i(),t("div",null,a[0]||(a[0]=[e(`<h2 id="How-to-Install-Reactant.jl?" tabindex="-1">How to Install Reactant.jl? <a class="header-anchor" href="#How-to-Install-Reactant.jl?" aria-label="Permalink to &quot;How to Install Reactant.jl? {#How-to-Install-Reactant.jl?}&quot;">​</a></h2><p>Its easy to install Reactant.jl. Since Reactant.jl is registered in the Julia General registry, you can simply run the following command in the Julia REPL:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;Reactant&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>If you want to use the latest unreleased version of Reactant.jl, you can run the following command:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(url</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;https://github.com/EnzymeAD/Reactant.jl&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><h2 id="Select-an-Accelerator-Backend" tabindex="-1">Select an Accelerator Backend <a class="header-anchor" href="#Select-an-Accelerator-Backend" aria-label="Permalink to &quot;Select an Accelerator Backend {#Select-an-Accelerator-Backend}&quot;">​</a></h2><div class="vp-code-group vp-adaptive-theme"><div class="tabs"><input type="radio" name="group-pINbk" id="tab-j0WsYbr" checked><label data-title="CPUs" for="tab-j0WsYbr">CPUs</label><input type="radio" name="group-pINbk" id="tab-dtxzoOY"><label data-title="NVIDIA GPUs" for="tab-dtxzoOY">NVIDIA GPUs</label><input type="radio" name="group-pINbk" id="tab-fidZ5Mb"><label data-title="Cloud TPUs" for="tab-fidZ5Mb">Cloud TPUs</label></div><div class="blocks"><div class="language-julia vp-adaptive-theme active"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">set_default_backend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;cpu&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">set_default_backend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;gpu&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Reactant</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Reactant</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">set_default_backend</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;tpu&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div></div></div>`,7)]))}const g=s(l,[["render",n]]);export{r as __pageData,g as default};