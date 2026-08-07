/* eslint-disable @typescript-eslint/no-require-imports */
const path = require("path");

const dev = process.env.NODE_ENV !== "production";

module.exports = {
  presets: ["next/babel"],
  plugins: [
    [
      "@stylexjs/babel-plugin",
      {
        dev,
        runtimeInjection: false,
        enableInlinedConditionalMerge: true,
        treeshakeCompensation: true,
        aliases: {
          "@/*": [path.join(__dirname, "src", "*")],
        },
        // NOTE: keep the default `themeFileExtension` (`.stylex`). The plugin
        // matches import specifiers ending in `.stylex` (e.g. `./tokens.stylex`)
        // and then resolves the concrete `.stylex.ts`/`.stylex.js` file itself.
        unstable_moduleResolution: { type: "commonJS" },
      },
    ],
  ],
};


