import{_ as a,c as s,o as i,a2 as t}from"./chunks/framework.DpQpLFOt.js";const u=JSON.parse('{"title":"Local build of ReactantExtra","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/local-build.md","filePath":"tutorials/local-build.md","lastUpdated":null}'),o={name:"tutorials/local-build.md"};function l(n,e,r,c,d,p){return i(),s("div",null,e[0]||(e[0]=[t(`<h1 id="local-build" tabindex="-1">Local build of ReactantExtra <a class="header-anchor" href="#local-build" aria-label="Permalink to &quot;Local build of ReactantExtra {#local-build}&quot;">​</a></h1><p>In the <code>deps/</code> subdirectory of the Reactant repository there is a script to do local builds of ReactantExtra, including debug builds.</p><h2 id="requirements" tabindex="-1">Requirements <a class="header-anchor" href="#requirements" aria-label="Permalink to &quot;Requirements&quot;">​</a></h2><ul><li><p>Julia. If you don&#39;t have it already, you can obtain it from the <a href="https://julialang.org/downloads/" target="_blank" rel="noreferrer">official Julia website</a></p></li><li><p>A reasonably recent C/C++ compiler, ideally GCC 12+. Older compilers may not work.</p></li><li><p>Bazel. If you don&#39;t have it already, you can download a build for your platform from <a href="https://github.com/bazelbuild/bazelisk/releases/latest" target="_blank" rel="noreferrer">the latest <code>bazelbuild/bazelisk</code> release</a> and put the <code>bazel</code> executable in <code>PATH</code></p></li><li><p>not necessary in general, but for debug builds with CUDA support, you&#39;ll need a fast linker, like <code>lld</code> or <code>mold</code> Binutils <code>ld</code> won&#39;t work, don&#39;t even try using it. You can obtain <code>mold</code> for your platform from the <a href="https://github.com/rui314/mold/releases/latest" target="_blank" rel="noreferrer">latest <code>rui314/mold</code> release</a> and put the <code>mold</code> executable in <code>PATH</code></p></li></ul><h2 id="building" tabindex="-1">Building <a class="header-anchor" href="#building" aria-label="Permalink to &quot;Building&quot;">​</a></h2><p>At a high-level, after you <code>cd</code> to the <code>deps/</code> directory you can run the commands</p><div class="language-bash vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">julia</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --project</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -e</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &#39;using Pkg; Pkg.instantiate()&#39;</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> # needed only the first time to install dependencies for this script</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">julia</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> -O0</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --color=yes</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> --project</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> build_local.jl</span></span></code></pre></div><p>There are a few of options you may want to use to tweak the build. For more information run the command (what&#39;s show below may not be up to date, run the command locally to see the options available to you):</p><div class="language-console vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">console</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">% julia -O0 --project build_local.jl --help</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">usage: build_local.jl [--debug] [--backend BACKEND]</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                      [--gcc_host_compiler_path GCC_HOST_COMPILER_PATH]</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                      [--cc CC]</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                      [--hermetic_python_version HERMETIC_PYTHON_VERSION]</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                      [--jobs JOBS] [--copt COPT] [--cxxopt CXXOPT]</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                      [--extraopt EXTRAOPT] [--color COLOR] [-h]</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">optional arguments:</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --debug               Build with debug mode (-c dbg).</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --backend BACKEND     Build with the specified backend (auto, cpu,</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                        cuda). (default: &quot;auto&quot;)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --gcc_host_compiler_path GCC_HOST_COMPILER_PATH</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                        Path to the gcc host compiler. (default:</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                        &quot;/usr/bin/gcc&quot;)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --cc CC                (default: &quot;/usr/bin/cc&quot;)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --hermetic_python_version HERMETIC_PYTHON_VERSION</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                        Hermetic Python version. (default: &quot;3.10&quot;)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --jobs JOBS           Number of parallel jobs. (type: Int64,</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                        default: &lt;MAXIMUM NUMBER OF CPUs&gt;)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --copt COPT           Options to be passed to the C compiler.  Can</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                        be used multiple times.</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --cxxopt CXXOPT       Options to be passed to the C++ compiler.  Can</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                        be used multiple times.</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --extraopt EXTRAOPT   Extra options to be passed to Bazel.  Can be</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                        used multiple times.</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  --color COLOR         Set to \`yes\` to enable color output, or \`no\`</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                        to disable it. Defaults to same color setting</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                        as the Julia process. (default: &quot;no&quot;)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">  -h, --help            show this help message and exit</span></span></code></pre></div><h3 id="Doing-a-build-on-a-system-with-memoryor-number-of-processes-restrictions" tabindex="-1">Doing a build on a system with memoryor number of processes restrictions <a class="header-anchor" href="#Doing-a-build-on-a-system-with-memoryor-number-of-processes-restrictions" aria-label="Permalink to &quot;Doing a build on a system with memoryor number of processes restrictions {#Doing-a-build-on-a-system-with-memoryor-number-of-processes-restrictions}&quot;">​</a></h3><p>If you try to do the build on certain systems where there are in place restrictions on the number of processes or memory that your user can use (for example login node of clusters), you may have to limit the number of parallel jobs used by Bazel. By default Bazel would try to use the maximum number of CPUs available on the system, if you need reduce that pass the <code>--jobs JOBS</code> flag option. If the Bazel server is terminated abruptly with an error which looks like</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Server terminated abruptly (error code: 14, error message: &#39;Socket closed&#39;, log file: &#39;/path/to/server/jvm.out&#39;)</span></span></code></pre></div><p>that may be due to the fact that the build process was using too many resources (e.g. if concurrent compiler processes are cumulatively using too much memory or too many threads). Also in this case reducing the number of parallel jobs may be beneficial.</p><h3 id="CUDA-debug-build" tabindex="-1">CUDA debug build <a class="header-anchor" href="#CUDA-debug-build" aria-label="Permalink to &quot;CUDA debug build {#CUDA-debug-build}&quot;">​</a></h3><p>A CUDA debug build (<code>--debug --backend=cuda</code>) requires a recent GCC compiler (at least v12) and also a fast linker (see requirements above). You can tell GCC to use either <code>lld</code> or <code>mold</code> with <code>--extraopt &#39;--linkopt=-fuse-ld=lld&#39;</code> or <code>--extraopt &#39;--linkopt=-fuse-ld=mold&#39;</code> respectively. NOTE: the option <code>-fuse-ld=mold</code> was added in GCC 12, if you&#39;re trying to use an older version you can have some luck by making a symlink named <code>ld</code> pointing to <code>mold</code> in <code>PATH</code>, with higher precendce than Binutils <code>ld</code>.</p><h3 id="Optimised-build-with-debug-symbols" tabindex="-1">Optimised build with debug symbols <a class="header-anchor" href="#Optimised-build-with-debug-symbols" aria-label="Permalink to &quot;Optimised build with debug symbols {#Optimised-build-with-debug-symbols}&quot;">​</a></h3><p>Unoptimised builds of Reactant can be <em>extremely</em> slow at runtime. You may have more luck by doing an optimised build but retain (don&#39;t strip) the debug symbols, which in Bazel can achieved with the options <code>--strip=never --copt -g -c opt</code>. To do that using this <code>build_local.jl</code> script pass the options <code>--extraopt &#39;--strip=never&#39; --copt -g</code> (optimised builds are the default, unless you use <code>--debug</code>).</p><h3 id="Using-ccache" tabindex="-1">Using ccache <a class="header-anchor" href="#Using-ccache" aria-label="Permalink to &quot;Using ccache {#Using-ccache}&quot;">​</a></h3><p>If you want to use <code>ccache</code> as your compiler, you may have to add the flag <code>--extraopt &quot;--sandbox_writable_path=/path/to/ccache/directory&quot;</code> to let <code>ccache</code> write to its own directory.</p><h2 id="LocalPreferences.toml-file" tabindex="-1"><code>LocalPreferences.toml</code> file <a class="header-anchor" href="#LocalPreferences.toml-file" aria-label="Permalink to &quot;\`LocalPreferences.toml\` file {#LocalPreferences.toml-file}&quot;">​</a></h2><p>At the end of a successfult build, the <code>build_local.jl</code> script will create a <code>LocalPreferences.toml</code> file (see <a href="https://juliapackaging.github.io/Preferences.jl/stable/" target="_blank" rel="noreferrer"><code>Preferences.jl</code> documentation</a> for more information) in the top-level of the Reactant repository, pointing <code>libReactantExtra</code> to the new local build. If you instantiate this environment Reactant will automatically use the new local build, but if you want to use the local build in a different environment you will have to copy the <code>LocalPreferences.toml</code> file (or its content, if you already have a <code>LocalPreferences.toml</code> file) to the directory of that environment.</p>`,21)]))}const b=a(o,[["render",l]]);export{u as __pageData,b as default};
