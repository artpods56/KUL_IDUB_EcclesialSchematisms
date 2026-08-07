import { copyFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const workspace = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = resolve(
  workspace,
  "node_modules/maplibre-gl/dist/maplibre-gl-csp-worker.js",
);
const destination = resolve(workspace, "public/maplibre-gl-csp-worker.js");

await mkdir(dirname(destination), { recursive: true });
await copyFile(source, destination);
