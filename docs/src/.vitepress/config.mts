import { defineConfig } from "vitepress";
import { tabsMarkdownPlugin } from "vitepress-plugin-tabs";
import mathjax3 from "markdown-it-mathjax3";
import footnote from "markdown-it-footnote";
import { transformerMetaWordHighlight } from "@shikijs/transformers";

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

  head: [
    ["link", { rel: "icon", href: "REPLACE_ME_DOCUMENTER_VITEPRESS_FAVICON" }],
    ["script", { src: `/versions.js` }],
    ["script", { src: `${baseTemp.base}siteinfo.js` }],
  ],

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
      { text: "Getting Started", link: "/introduction" },
      { text: "Benchmarks", link: "https://enzymead.github.io/Reactant.jl/benchmarks/" },
      { text: "Tutorials", link: "/tutorials/" },
      {
        text: "API",
        items: [
          { text: "Core Reactant API", link: "/api/api" },
          {
            text: "MLIR Dialects",
            items: [
              { text: "ArithOps", link: "/api/arith" },
              { text: "Affine", link: "/api/affine" },
              { text: "Builtin", link: "/api/builtin" },
              { text: "Chlo", link: "/api/chlo" },
              { text: "Enzyme", link: "/api/enzyme" },
              { text: "Func", link: "/api/func" },
              { text: "StableHLO", link: "/api/stablehlo" },
              { text: "VHLO", link: "/api/vhlo" },
            ],
          },
          {
            text: "Low-Level API",
            items: [
              { text: "MLIR API", link: "/api/mlirc" },
              { text: "XLA", link: "/api/xla" },
            ],
          }
        ],
      },
      {
        component: "VersionPicker",
      },
    ],
    sidebar: {
      "/introduction/": {
        // @ts-ignore
        text: "Getting Started",
        collapsed: false,
        items: [
          { text: "Introduction", link: "/introduction" },
        ],
      },
      "/tutorials/": {
        text: "Tutorials",
        collapsed: false,
        items: [
          { text: "Overview", link: "/tutorials/" },
        ],
      },
      "/api/": {
        text: "API Reference",
        collapsed: false,
        items: [
          {
            text: "Reactant API",
            link: "/api/api",
          },
          {
            text: "MLIR Dialects",
            collapsed: false,
            items: [
              { text: "ArithOps", link: "/api/arith" },
              { text: "Affine", link: "/api/affine" },
              { text: "Builtin", link: "/api/builtin" },
              { text: "Chlo", link: "/api/chlo" },
              { text: "Enzyme", link: "/api/enzyme" },
              { text: "Func", link: "/api/func" },
              { text: "StableHLO", link: "/api/stablehlo" },
              { text: "VHLO", link: "/api/vhlo" },
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
        ],
      },
    },
    editLink: {
      pattern: "https://github.com/EnzymeAD/Reactant.jl/edit/main/docs/src/:path",
      text: "Edit this page on GitHub",
    },
    socialLinks: [
      { icon: "github", link: "https://github.com/EnzymeAD/Reactant.jl" },
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
