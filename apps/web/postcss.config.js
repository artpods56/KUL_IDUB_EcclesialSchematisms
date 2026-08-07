/* eslint-disable @typescript-eslint/no-require-imports */
const path = require("path");
const babelConfig = require("./babel.config");

const APP_ROOT = __dirname;

module.exports = {
  plugins: {
    "@stylexjs/postcss-plugin": {
      cwd: APP_ROOT,
      include: [path.join(APP_ROOT, "src/**/*.{js,jsx,ts,tsx}")],
      babelConfig: {
        babelrc: false,
        parserOpts: { plugins: ["typescript", "jsx"] },
        plugins: babelConfig.plugins,
      },
      useCSSLayers: true,
    },
    autoprefixer: {},
  },
};

