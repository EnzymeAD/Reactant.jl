import { defineConfig } from "vitepress";
import { tabsMarkdownPlugin } from "vitepress-plugin-tabs";
import mathjax3 from "markdown-it-mathjax3";
import footnote from "markdown-it-footnote";
import { transformerMetaWordHighlight } from "@shikijs/transformers";
import path from 'path'

const baseTemp = {
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS', // TODO: replace this in makedocs!
};

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  title: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  description: "Documentation for Reactant.jl",
  cleanUrls: true,
  outDir: 'REPLACE_ME_DOCUMENTER_VITEPRESS', // This is required for MarkdownVitepress to work correctly...
  ignoreDeadLinks: false, // tested in Documenter.jl directly
  lastUpdated: true,
  
  head: [
    ['link', { rel: "icon", href: `${baseTemp.base}favicon.ico` }],
    ['script', {src: `/versions.js` }],
    ['script', { src: `${baseTemp.base}siteinfo.js` }],
  ],
  vite: {
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '../components')
      }
    },
    optimizeDeps: {
      exclude: [ 
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ], 
    }, 
    ssr: { 
      noExternal: [ 
        // If there are other packages that need to be processed by Vite, you can add them here.
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ], 
    },
  },
  markdown: {
    math: true,
    config(md) {
      md.use(tabsMarkdownPlugin), md.use(mathjax3), md.use(footnote);
    },
    theme: {
      light: "github-light",
      dark: "github-dark",
    },
    codeTransformers: [transformerMetaWordHighlight()],
  },

  themeConfig: {
    outline: "deep",
    // https://vitepress.dev/reference/default-theme-config
    logo: {
      light: "/logo.svg",
      dark: "/logo.svg",  // XXX: different logo for dark mode???
    },
    search: {
      provider: "local",
      options: {
        detailedView: true,
      },
    },
    nav: [
      { text: "Home", link: "/" },
      { text: "Getting Started",
        items: [
          { text: "Introduction", link: "/introduction" },
          { text: "Configuration", link: "/introduction/configuration" },
        ],
      },
      { text: "Benchmarks", link: "https://enzymead.github.io/Reactant.jl/benchmarks/" },
      {
        text: "Tutorials",
        items: [
          {text: "Overview", link: "/tutorials/"},
          {text: "Profiling", link: "/tutorials/profiling"},
          {text: "Distributed", link: "/tutorials/multihost"},
          {text: "Local build", link: "/tutorials/local-build"},
        ],
      },
      {
        text: "API",
        items: [
          { text: "Core Reactant API", link: "/api/api" },
          { text: "Sharding", link: "/api/sharding" },
          { text: "Ops", link: "/api/ops" },
          {
            text: "MLIR Dialects",
            items: [
              { text: "ArithOps", link: "/api/dialects/arith" },
              { text: "Affine", link: "/api/dialects/affine" },
              { text: "Builtin", link: "/api/dialects/builtin" },
              { text: "Chlo", link: "/api/dialects/chlo" },
              { text: "Enzyme", link: "/api/dialects/enzyme" },
              { text: "EnzymeXLA", link: "/api/dialects/enzymexla" },
              { text: "Func", link: "/api/dialects/func" },
              { text: "GPU", link: "/api/dialects/gpu" },
              { text: "LLVM", link: "/api/dialects/llvm" },
              { text: "MPI", link: "/api/dialects/mpi" },
              { text: "MemRef", link: "/api/dialects/memref" },
              { text: "NVVM", link: "/api/dialects/nvvm" },
              { text: "Shardy", link: "/api/dialects/shardy" },
              { text: "StableHLO", link: "/api/dialects/stablehlo" },
              { text: "Triton", link: "/api/dialects/triton" },
              { text: "TPU", link: "/api/dialects/tpu" },
              { text: "VHLO", link: "/api/dialects/vhlo" },
            ],
          },
          {
            text: "Low-Level API",
            items: [
              { text: "MLIR API", link: "/api/mlirc" },
              { text: "XLA", link: "/api/xla" },
            ],
          },
          { text: "Internal API", link: "/api/internal" },
        ],
      },
      {
        component: "VersionPicker",
      },
    ],
    sidebar: {
      "/introduction/": [
        {
        text: "Getting Started",
        collapsed: false,
        items: [
          { text: "Introduction", link: "/introduction" },
          { text: "Configuration", link: "/introduction/configuration" },
        ],
      }
    ],
      "/tutorials/": [
        {
        text: "Tutorials",
        collapsed: false,
        items: [
          { text: "Overview", link: "/tutorials/" },
          { text: "Profiling", link: "/tutorials/profiling" },
          { text: "Distributed", link: "/tutorials/multihost" },
          { text: "Local build", link: "/tutorials/local-build" },
        ],
      }
    ],
      "/api/": [
        {
        text: "API Reference",
        collapsed: false,
        items: [
          {
            text: "Reactant API",
            link: "/api/api",
          },
          { text: "Sharding", link: "/api/sharding" },
          { text: "Ops", link: "/api/ops" },
          {
            text: "MLIR Dialects",
            collapsed: false,
            items: [
              { text: "ArithOps", link: "/api/dialects/arith" },
              { text: "Affine", link: "/api/dialects/affine" },
              { text: "Builtin", link: "/api/dialects/builtin" },
              { text: "Chlo", link: "/api/dialects/chlo" },
              { text: "Enzyme", link: "/api/dialects/enzyme" },
              { text: "EnzymeXLA", link: "/api/dialects/enzymexla" },
              { text: "Func", link: "/api/dialects/func" },
              { text: "GPU", link: "/api/dialects/gpu" },
              { text: "LLVM", link: "/api/dialects/llvm" },
              { text: "MPI", link: "/api/dialects/mpi" },
              { text: "MemRef", link: "/api/dialects/memref" },
              { text: "NVVM", link: "/api/dialects/nvvm" },
              { text: "Shardy", link: "/api/dialects/shardy" },
              { text: "StableHLO", link: "/api/dialects/stablehlo" },
              { text: "Triton", link: "/api/dialects/triton" },
              { text: "TPU", link: "/api/dialects/tpu" },
              { text: "VHLO", link: "/api/dialects/vhlo" },
            ],
          },
          {
            text: "Low-Level API",
            collapsed: false,
            items: [
              { text: "MLIR API", link: "/api/mlirc" },
              { text: "XLA", link: "/api/xla" },
            ],
          },
          { text: "Internal API", link: "/api/internal" },
        ],
      }
    ],
    },
    editLink: {
      pattern: "https://github.com/EnzymeAD/Reactant.jl/edit/main/docs/src/:path",
      text: "Edit this page on GitHub",
    },
    socialLinks: [
      { icon: "slack", link: "https://julialang.org/slack/" },
    ],
    footer: {
      message:
        'Made with <a href="https://documenter.juliadocs.org/stable/" target="_blank"><strong>Documenter.jl</strong></a>, <a href="https://vitepress.dev" target="_blank"><strong>VitePress</strong></a> and <a href="https://luxdl.github.io/DocumenterVitepress.jl/stable" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>Released under the MIT License. Powered by the <a href="https://www.julialang.org">Julia Programming Language</a>.<br>',
      copyright: `© Copyright ${new Date().getUTCFullYear()} Reactant Development Team.`,
    },
    lastUpdated: {
      text: "Updated at",
      formatOptions: {
        dateStyle: "full",
        timeStyle: "medium",
      },
    },
  },
});
