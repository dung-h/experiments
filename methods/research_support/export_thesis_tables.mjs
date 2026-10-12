// Plain-CSV thesis tables authored through bundled Artifact Tool typed ranges.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
const require = createRequire('/home/server/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/_thesis_loader.cjs');
const { Workbook } = await import(require.resolve('@oai/artifact-tool'));
let text = '';
for await (const chunk of process.stdin) text += chunk;
const payload = JSON.parse(text);
const expected = path.resolve('artifacts/thesis_synthesis/v1/canonical_tables');
if (path.resolve(payload.output_dir) !== expected) throw new Error('output scope mismatch');
await fs.mkdir(expected, { recursive: true });
const escape = value => {
  const s = value === null ? '' : String(value);
  return /[",\r\n]/.test(s) ? '"' + s.replaceAll('"', '""') + '"' : s;
};
for (const [name, rows] of Object.entries(payload.tables)) {
  if (!/^[a-h]_[a-z_]+\.csv$/.test(name)) throw new Error('non-canonical filename');
  const headers = Object.keys(rows[0]);
  const matrix = [headers, ...rows.map(row => headers.map(key => row[key]))];
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add('Results');
  const range = sheet.getRangeByIndexes(0, 0, matrix.length, headers.length);
  range.values = matrix;
  workbook.recalculate();
  const restored = range.values;
  if (JSON.stringify(restored) !== JSON.stringify(matrix)) throw new Error(`typed-value drift: ${name}`);
  const csv = restored.map(row => row.map(escape).join(',')).join('\n') + '\n';
  await fs.writeFile(path.join(expected, name), csv, 'utf8');
}
process.stdout.write(JSON.stringify({status:'PASS',tables:Object.keys(payload.tables).length,format:'CSV',formula_evaluation:'not applicable; frozen saved statistics'}));
