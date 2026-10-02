import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "../..");
// Rebuilds stay outside the committed snapshot. Existing exports are never overwritten.
const workspaceDir = path.join(repoRoot, "work/presentation-deck");
const scorecardDir = path.join(
  repoRoot,
  "artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3",
);
const figureDir = path.join(repoRoot, "presentations/figures");
const buildDir = path.join(workspaceDir, ".build");
const outputDir = path.join(workspaceDir, "output");
const finalPptx = path.join(outputDir, "quantum_runtime_benchmark_vi.pptx");
const skillDir = process.env.SKILL_DIR;
try {
  await fs.access(finalPptx);
  throw new Error(`Refusing to overwrite ${finalPptx}; choose a fresh build workspace.`);
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}

if (!path.isAbsolute(skillDir ?? "")) {
  throw new Error("Set SKILL_DIR to the installed Presentations skill directory.");
}

const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } =
  await import(pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href);

const C = {
  ink: "#163444",
  ink2: "#25495A",
  teal: "#007F78",
  tealPale: "#E4F2F0",
  amber: "#B66A18",
  amberPale: "#FBF0E1",
  muted: "#526873",
  pale: "#F4F7F8",
  line: "#CBD6DA",
  white: "#FFFFFF",
};
const W = 1280;
const H = 720;
// Noto Sans is present in the official runtime and covers Vietnamese text.
// Pin it explicitly, including the presentation theme used by native tables.
const fontFamily = resolvePresentationFont({ fontFamily: "Noto Sans" });
const presentation = Presentation.create({ slideSize: { width: W, height: H } });
presentation.theme.defaultFont = fontFamily;

async function requireFigures() {
  const files = {
    qpu: "fig1_qpu_paired_mae_by_source.png",
    coverage: "fig2_qpu_prediction_coverage.png",
    mps: "fig3_cudaq_mps_quality_gated_clocks.png",
    maestro: "fig4_maestro_pilot_timing_spread.png",
  };
  const missing = [];
  for (const filename of Object.values(files)) {
    try {
      await fs.access(path.join(figureDir, filename));
    } catch {
      missing.push(filename);
    }
  }
  if (missing.length) {
    throw new Error(`F-C4 figure bundle is incomplete at ${figureDir}: ${missing.join(", ")}`);
  }
  return files;
}

function addText(slide, text, position, style = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    typeface: fontFamily,
    fontSize: 23,
    color: C.ink,
    autoFit: "none",
    ...style,
  };
  return box;
}

function newSlide(title, index, notes, options = {}) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  if (options.cover || options.fullFigure) {
    slide.speakerNotes.textFrame.setText(notes);
    return slide;
  }
  addText(slide, title, { left: 64, top: 38, width: 1152, height: 66 }, {
    fontSize: 43,
    bold: true,
    color: C.ink,
  });
  if (!options.noFooter) {
    addText(slide, String(index).padStart(2, "0"), { left: 1160, top: 668, width: 55, height: 28 }, {
      fontSize: 16,
      color: C.muted,
      alignment: "right",
    });
  }
  slide.speakerNotes.textFrame.setText(notes);
  return slide;
}

function addTable(slide, values, position, opts = {}) {
  const rows = values.length;
  const columns = values[0].length;
  const table = slide.tables.add({
    rows,
    columns,
    left: position.left,
    top: position.top,
    width: position.width,
    height: position.height,
    ...(opts.columnWidths ? { columnWidths: opts.columnWidths } : {}),
    values,
  });
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  const bodyRange = table.cells.block({ row: 0, column: 0, rowCount: rows, columnCount: columns });
  bodyRange.textStyle.typeface = fontFamily;
  bodyRange.textStyle.fontSize = opts.fontSize ?? 20;
  bodyRange.textStyle.color = C.ink;
  bodyRange.textStyle.bold = false;
  for (let row = 0; row < rows; row += 1) {
    table.rows[row].height = position.height / rows;
    for (let column = 0; column < columns; column += 1) {
      table.getCell(row, column).fill = row === 0
        ? C.ink2
        : (row % 2 === 0 ? C.pale : C.white);
    }
  }
  const header = table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: columns });
  header.textStyle.typeface = fontFamily;
  header.textStyle.fontSize = opts.headerFontSize ?? (opts.fontSize ?? 20);
  header.textStyle.color = C.white;
  header.textStyle.bold = true;
  return table;
}

async function addFigure(slide, filename, alt, position) {
  const bytes = new Uint8Array(await fs.readFile(path.join(figureDir, filename)));
  slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt,
    fit: "contain",
    position,
  });
}

function addImageCaption(slide, caption, position) {
  addText(slide, caption, position, {
    fontSize: 17,
    color: C.muted,
  });
}

function addBody(slide, text, position, style = {}) {
  return addText(slide, text, position, { fontSize: 24, ...style });
}

const figures = await requireFigures();
await fs.access(path.join(figureDir, "README.md"));

// 1. Cover and scope.
{
  const slide = newSlide("", 1, [
    "Báo cáo benchmark nội bộ, khoảng 10 phút.",
    `Evidence: ${path.relative(repoRoot, path.join(scorecardDir, "REPORT.md"))}`,
    `Brief authority: ${path.relative(repoRoot, path.join(repoRoot, "docs/presentation.md"))}`,
  ].join("\n"), { cover: true });
  addText(slide, "Ước lượng runtime lượng tử", { left: 74, top: 115, width: 1090, height: 88 }, {
    fontSize: 60,
    bold: true,
    color: C.ink,
  });
  addText(slide, "Bằng chứng từ QPU lưu trữ và simulator local", { left: 78, top: 222, width: 1050, height: 55 }, {
    fontSize: 32,
    color: C.teal,
  });
  addText(slide, "Phạm vi khoa học PARTIAL", { left: 78, top: 358, width: 900, height: 52 }, {
    fontSize: 30,
    bold: true,
    color: C.amber,
  });
  addBody(slide, "Không chạy live QPU. Không gộp các clock khác nhau thành một bảng xếp hạng.", {
    left: 78, top: 426, width: 1050, height: 76,
  }, { fontSize: 25 });
  addText(slide, "Bản nội bộ | 02.10.2026", { left: 78, top: 638, width: 400, height: 30 }, {
    fontSize: 18,
    color: C.muted,
  });
}

// 2. Population and clocks.
{
  const slide = newSlide("Dữ liệu và phạm vi", 2,
    `Nguồn: ${path.relative(repoRoot, path.join(scorecardDir, "REPORT.md"))}\n` +
    `Chi tiết QPU: ${path.relative(repoRoot, path.join(scorecardDir, "qpu/archived_method_coverage.csv"))}\n` +
    `Panel simulator: ${path.relative(repoRoot, path.join(scorecardDir, "simulator/common_exact_qasm_panel.csv"))}\n` +
    "QPU nguồn có 340 Ma–Li, 4.482 Qonductor và 3.945 QPack observations. Simulator gồm 204 members, 191 exact hashes, 162 core và 42 frontier. QPU observed service, QPU scheduled duration và local simulator execution là các clock riêng.");
  addTable(slide, [
    ["Miền", "Đơn vị dữ liệu", "Phạm vi"],
    ["QPU Ma–Li", "340 observations", "Nguồn riêng"],
    ["QPU Qonductor", "4.482 observations", "Nguồn riêng"],
    ["QPU QPack", "3.945 observations", "46 workflows"],
    ["Tổng QPU", "8.767 observations", "Labels archived"],
    ["Simulator local", "204 members / 191 hashes", "162 core + 42 frontier"],
  ], { left: 70, top: 145, width: 1140, height: 410 }, { fontSize: 22, headerFontSize: 22, columnWidths: [230, 370, 540] });
  addBody(slide, "Đây là hai domain với target và clock riêng, không phải một leaderboard phẳng.", {
    left: 76, top: 580, width: 1100, height: 50,
  }, { fontSize: 23, bold: true, color: C.teal });
}

// 3. Comparison design.
{
  const slide = newSlide("Thiết kế so sánh", 3,
    `Nguồn: ${path.relative(repoRoot, path.join(repoRoot, "docs/presentation.md"))}\n` +
    `Scorecard authority: ${path.relative(repoRoot, path.join(scorecardDir, "manifest.json"))}\n` +
    `QPU coverage denominator: ${path.relative(repoRoot, path.join(scorecardDir, "qpu/archived_method_coverage.csv"))}`);
  addTable(slide, [
    ["Nguyên tắc", "Áp dụng", "Giới hạn"],
    ["Rows chung", "Graph V3-large và polynomial dùng cùng unified ledger và grouped outer folds", "So sánh theo source-local shared rows"],
    ["Fit", "Preprocessing và calibration chỉ fit trên outer-train", "Không chọn theo labels của test"],
    ["Coverage", "Unavailable và failures giữ trong denominator được giao", "Coverage khác accuracy"],
    ["Clock", "Hiển thị output clock cùng evaluation target clock", "Không xếp hạng chéo clock"],
  ], { left: 70, top: 160, width: 1140, height: 380 }, { fontSize: 20, headerFontSize: 21, columnWidths: [230, 475, 435] });
  addBody(slide, "MAE thấp hơn trên shared rows không tự động chứng minh superiority trên toàn ledger.", {
    left: 76, top: 580, width: 1100, height: 52,
  }, { fontSize: 24, bold: true, color: C.teal });
}

// 4. Coverage across methods.
{
  const slide = newSlide("Coverage phương pháp", 4,
    `Method scope: ${path.relative(repoRoot, path.join(scorecardDir, "method_scope.csv"))}\n` +
    `QPU coverage table: ${path.relative(repoRoot, path.join(scorecardDir, "qpu/archived_method_coverage.csv"))}\n` +
    `Full 24-variant detail: ${path.relative(repoRoot, path.join(figureDir, figures.coverage))}; metadata in ${path.relative(repoRoot, path.join(figureDir, "README.md"))}`);
  addTable(slide, [
    ["QPU approach", "Evidence status", "Simulator approach", "Evidence status"],
    ["Ma–Li graph", "Unified adaptation; V3 and V3-large", "Ma–Li local graph", "162 core rows; supplementary diagnostic"],
    ["Qonductor", "Unified polynomial adaptation", "Azizov", "Classical adaptations only"],
    ["Scholten", "Nominal proxy; 5,769 / 8,767", "Family-aware", "Local joint task not evaluated"],
    ["QCRE", "Scheduled proxy; 8,766 / 8,767", "Maestro", "Pilot gate failed; no predictor score"],
    ["Qiskit estimate_duration", "Native schedule; 8,766 / 8,767", "cuTensorNet", "Same-plan estimate; 603 / 612 ok"],
    ["Hyb-HANAS", "One-circuit cost adaptation; 8,420 finite", "Pasqal emu-MPS", "Analog pilot not promoted"],
  ], { left: 64, top: 148, width: 1152, height: 450 }, { fontSize: 17, headerFontSize: 18, columnWidths: [205, 345, 210, 392] });
  addBody(slide, "24 QPU variants are not 24 papers. Coverage is not accuracy; output clocks differ.", {
    left: 70, top: 615, width: 1125, height: 38,
  }, { fontSize: 21, color: C.teal, bold: true });
}

// 5. Paired QPU error.
{
  const slide = newSlide("MAE theo source trên shared rows", 5,
    `Figure: ${path.relative(repoRoot, path.join(figureDir, figures.qpu))}\n` +
    `Figure caption/metadata: ${path.relative(repoRoot, path.join(figureDir, "README.md"))}\n` +
    `Source table: ${path.relative(repoRoot, path.join(scorecardDir, "qpu/archived_method_pairwise_comparisons.csv"))}\n` +
    "Comparison: candidate_method_id=mali_graph_architecture_v3_large; reference_method_id=unified_polynomial_v3. Units: seconds of MAE difference, graph minus polynomial. Evaluation target: archived observed service/execution time. Shared-row hashes in source order: Ma–Li 3ad175f3b2cf88f52f9f0dedd404684fbff696b2d776f8d55538083dcdeac8e7; Qonductor a785df4abc98e96ad899bdf4f53837806715232d440bc620de74b88639c74655; QPack d8f5438afbc5179bc99a96672022627411c6d6ead8597fcee6b9abbe383a36d7. Ma–Li interval crosses zero. Qonductor graph coverage is asymmetric because row4477 is unavailable. This is paired descriptive evidence, not primary superiority on 4,482 rows.");
  await addFigure(slide, figures.qpu, "Paired MAE comparison by QPU source with grouped-bootstrap intervals", {
    left: 74, top: 142, width: 1132, height: 474,
  });
  addImageCaption(slide, "Shared rows by source. 95% grouped-bootstrap CI for graph minus polynomial MAE. Lower MAE favors the graph adaptation.", {
    left: 78, top: 620, width: 1120, height: 38,
  });
}

// 6. QPack multiple metrics.
{
  const slide = newSlide("QPack: MAE và R²", 6,
    `Source metrics: ${path.relative(repoRoot, path.join(scorecardDir, "qpu/archived_method_source_metrics.csv"))}\n` +
    `Unified pairwise results: ${path.relative(repoRoot, path.join(scorecardDir, "qpu/archived_method_pairwise_comparisons.csv"))}\n` +
    `Limitations: ${path.relative(repoRoot, path.join(repoRoot, "docs/presentation.md"))}\n` +
    "QPack slice is row-weighted for the primary displayed MAE and R². Equal-workflow MAE is a diagnostic over 46 workflows. Six templates recur across multiple workflows; workflow holdout does not mean unseen-template validation. Negative R² means squared error exceeds that of the test-set mean reference, which is not a deployable baseline fitted from test labels.");
  addText(slide, "MAE (s)", { left: 110, top: 150, width: 470, height: 38 }, { fontSize: 24, bold: true, color: C.teal });
  const maeChart = slide.charts.add("bar", {
    position: { left: 80, top: 190, width: 510, height: 330 },
    categories: ["Graph V3-large", "Polynomial"],
    series: [{ name: "MAE (s)", values: [0.943547, 1.221289], fill: C.teal }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 80 },
    hasLegend: false,
    xAxis: { min: 0, max: 1.4, numberFormatCode: "0.0", title: { text: "Seconds" }, majorGridlines: { style: "solid", fill: C.line, width: 1 } },
    yAxis: { textStyle: { fontSize: 18, fill: C.ink } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: 18, fill: C.ink, bold: true } },
  });
  applyPresentationChartFont(maeChart, { fontFamily });
  addText(slide, "R²", { left: 690, top: 150, width: 470, height: 38 }, { fontSize: 24, bold: true, color: C.teal });
  // A native-shape score plot keeps negative-value labels in a fixed value
  // column; a native negative bar chart puts the category axis at zero and
  // collides its out-end label with the short Graph V3-large bar.
  const r2Plot = { left: 805, right: 1045, top: 232, bottom: 500 };
  const r2Ticks = [-0.8, -0.6, -0.4, -0.2, 0];
  for (const tick of r2Ticks) {
    const x = r2Plot.left + ((tick + 0.8) / 0.8) * (r2Plot.right - r2Plot.left);
    slide.shapes.add({
      geometry: "rect",
      position: { left: x, top: r2Plot.top, width: tick === 0 ? 2 : 1, height: r2Plot.bottom - r2Plot.top },
      fill: tick === 0 ? C.line : "#E7ECEE",
      line: { fill: "none", width: 0 },
    });
    addText(slide, tick === 0 ? "0.0" : tick.toFixed(1), {
      left: x - 24, top: 505, width: 48, height: 24,
    }, { fontSize: 16, color: C.muted, alignment: "center" });
  }
  const r2Rows = [
    { label: "Polynomial", value: -0.631629, valueText: "−0.631629", center: 276 },
    { label: "Graph V3-large", value: -0.094887, valueText: "−0.094887", center: 410 },
  ];
  for (const row of r2Rows) {
    const barStart = r2Plot.left + ((row.value + 0.8) / 0.8) * (r2Plot.right - r2Plot.left);
    addText(slide, row.label, { left: 650, top: row.center - 16, width: 142, height: 30 }, {
      fontSize: 18, color: C.muted, alignment: "right",
    });
    slide.shapes.add({
      geometry: "rect",
      position: { left: barStart, top: row.center - 35, width: r2Plot.right - barStart, height: 70 },
      fill: C.amber,
      line: { fill: "none", width: 0 },
    });
    addText(slide, row.valueText, { left: 1058, top: row.center - 16, width: 122, height: 30 }, {
      fontSize: 18, bold: true, color: C.ink,
    });
  }
  addBody(slide, "Equal-workflow MAE diagnostic: 0.946290 s vs 1.217754 s across 46 workflows. Both R² values remain negative.", {
    left: 78, top: 555, width: 1120, height: 60,
  }, { fontSize: 20, color: C.ink2 });
  addBody(slide, "6 templates recur; workflow holdout is not unseen-template validation.", {
    left: 78, top: 612, width: 1120, height: 38,
  }, { fontSize: 19, color: C.muted });
}

// 7. Clock boundaries for analytical methods.
{
  const slide = newSlide("Scheduled duration và observed service time", 7,
    `Method coverage: ${path.relative(repoRoot, path.join(scorecardDir, "qpu/archived_method_coverage.csv"))}\n` +
    `Method scope: ${path.relative(repoRoot, path.join(scorecardDir, "method_scope.csv"))}\n` +
    `Source notes: ${path.relative(repoRoot, path.join(repoRoot, "docs/presentation.md"))}\n` +
    "QCRE critical path and Qiskit estimate_duration outputs are scheduled single-shot proxies, not provider service/execution times. Their equality under the same Target is a cross-check, not two independent wins. Shot scaling and outer-train-only affine/log-affine calibration are adaptations.");
  addTable(slide, [
    ["Route", "Output clock", "What it excludes or changes"],
    ["QCRE critical path", "Scheduled single-shot", "No queue, service, transport, or fixed overhead"],
    ["Qiskit estimate_duration", "Scheduled single-shot", "Native API output under frozen Target"],
    ["Shot-scaled variants", "Scheduled total proxy", "Non-native multiplication by shots"],
    ["Outer-train calibration", "Calibrated service-time target", "Adaptation, fit on outer-train only"],
  ], { left: 70, top: 154, width: 1140, height: 360 }, { fontSize: 20, headerFontSize: 20, columnWidths: [280, 310, 550] });
  addBody(slide, "8.766 predictions per raw route. Row4477 is unavailable. Scheduled duration does not replace observed QPU service time.", {
    left: 78, top: 555, width: 1100, height: 60,
  }, { fontSize: 23, bold: true, color: C.teal });
}

// 8. Simulator execution and quality.
{
  const slide = newSlide("Simulator: coverage và quality gate", 8,
    `Figure: ${path.relative(repoRoot, path.join(figureDir, figures.mps))}\n` +
    `Figure captions/metadata: ${path.relative(repoRoot, path.join(figureDir, "README.md"))}\n` +
    `Availability: ${path.relative(repoRoot, path.join(scorecardDir, "simulator/availability_matrix.csv"))}\n` +
    `Per-configuration metrics: ${path.relative(repoRoot, path.join(scorecardDir, "simulator/per_configuration_metrics.csv"))}\n` +
    `Panel identities: ${path.relative(repoRoot, path.join(scorecardDir, "simulator/common_exact_qasm_panel.csv"))}\n` +
    "Counts are session-cells, not unique circuits. Keep engine, precision, first/warm clock, quality and failure states separate. CUDA-Q MPS quality failures remain execution attempts and are not dropped from coverage. cuTensorNet's selected-plan estimate is paired to same-plan contraction, not compared to full wall-clock of another simulator.");
  await addFigure(slide, figures.mps, "CUDA-Q MPS clock and quality-gate comparison, keeping first and warm sessions separate", {
    left: 70, top: 136, width: 1140, height: 426,
  });
  addBody(slide, "MPS: 612 session-cells per clock, 573 ok, 21 quality_failed, 18 adapter_error. TN: 612 sessions, 603 ok, 9 timeout.", {
    left: 74, top: 574, width: 1120, height: 38,
  }, { fontSize: 19, color: C.ink2 });
  addBody(slide, "Panel 204 members / 191 exact hashes. Failures remain in coverage; session-cells are not distinct circuits.", {
    left: 74, top: 615, width: 1120, height: 38,
  }, { fontSize: 19, color: C.teal, bold: true });
}

// 9. Maestro pilot stability.
{
  const slide = newSlide("", 9,
    `Figure: ${path.relative(repoRoot, path.join(figureDir, figures.maestro))}\n` +
    `Figure caption/metadata: ${path.relative(repoRoot, path.join(figureDir, "README.md"))}\n` +
    `Canonical pilot summary: ${path.relative(repoRoot, path.join(scorecardDir, "simulator/maestro_pilot_summary.csv"))}\n` +
    `Scope brief: ${path.relative(repoRoot, path.join(repoRoot, "docs/presentation.md"))}\n` +
    "Clock: maestro_qcsim_process_isolated_reported_execution. Fresh process per timing call; do not call these persistent-process warm measurements. Pilot terminal status: pilot_gate_failed; no resume. The 15 reported values belong to one statevector n=16, one-qubit noncommuting, R=256 cell and range from 0.014905151 s to 7.355305934 s. MPS n=12/chi=32 session medians: 0.033975028 / 0.033144007 / 0.035826843 s. Cause of variation is unknown; do not attribute it to threading, hardware, or algorithm. No panel attempt or predictor accuracy score exists.", { fullFigure: true });
  await addFigure(slide, figures.maestro, "Maestro pilot process-isolated reported execution timing spread", {
    left: 68, top: 0, width: 1144, height: 720,
  });
}

// 10. Takeaway and limitations.
{
  const slide = newSlide("Kết luận và phạm vi còn thiếu", 10,
    `Scorecard: ${path.relative(repoRoot, path.join(scorecardDir, "REPORT.md"))}\n` +
    `Method scope: ${path.relative(repoRoot, path.join(scorecardDir, "method_scope.csv"))}\n` +
    `Presentation brief: ${path.relative(repoRoot, path.join(repoRoot, "docs/presentation.md"))}`);
  addTable(slide, [
    ["Có thể kết luận", "Chưa đủ evidence để kết luận"],
    ["Graph V3-large có MAE thấp hơn polynomial trên source-local shared rows.", "Không có pooled cross-source superiority claim."],
    ["QPack có MAE thấp hơn nhưng cả hai R² đều âm.", "Không nói model giải thích tốt biến thiên runtime."],
    ["Simulator coverage có quality/failure gates rõ ràng.", "Run completed không đồng nghĩa accuracy benchmark hoàn tất."],
    ["Maestro pilot ghi nhận stability failure.", "Không có accepted predictor hoặc panel accuracy score."],
  ], { left: 70, top: 150, width: 1140, height: 390 }, { fontSize: 20, headerFontSize: 21, columnWidths: [550, 590] });
  addBody(slide, "Scientific coverage: PARTIAL. Build validation không đồng nghĩa full replication hoặc public-release approval.", {
    left: 78, top: 576, width: 1100, height: 70,
  }, { fontSize: 23, bold: true, color: C.amber });
}

// 11. Fidelity and reconstructed inputs.
{
  const slide = newSlide("Fidelity phương pháp và tái dựng đầu vào", 11,
    `Scorecard scope disclosures: ${path.relative(repoRoot, path.join(scorecardDir, "REPORT.md"))}\n` +
    `Method cards: ${path.relative(repoRoot, path.join(scorecardDir, "method_scope.csv"))}\n` +
    `Representation protocol: ${path.relative(repoRoot, path.join(repoRoot, "benchmark_v1/decisions/circuit_representation_v3.md"))}`);
  addTable(slide, [
    ["Mảng", "Đã dùng", "Giới hạn fidelity"],
    ["QPU graph", "Ma–Li/QPack logical; Qonductor exact submitted physical QASM", "Không thực thi compiled-input promise cho Ma–Li/QPack"],
    ["QPack replay", "6 reconstructed structures; representative rz(0.3)/rx(0.2)", "Không có original optimizer angles / submitted routing"],
    ["Graph features", "7 global features; DAG + global MLP branches", "Polynomial có 5 features; không cô lập causal graph effect"],
    ["Native estimates", "Snapshot calibration theo backend", "Không phải row-day historical calibration"],
    ["Unmeasured routes", "Family-aware, original Composer, joint quality", "Chưa có evaluated evidence phù hợp"],
  ], { left: 66, top: 148, width: 1148, height: 440 }, { fontSize: 18, headerFontSize: 19, columnWidths: [210, 490, 448] });
  addBody(slide, "Archived labels vẫn là runtime quan sát; replay reconstruction không thay đổi labels.", {
    left: 73, top: 612, width: 1115, height: 38,
  }, { fontSize: 20, color: C.teal, bold: true });
}

// 12. Clock, failure, and release semantics.
{
  const slide = newSlide("Clocks, quality, và release boundary", 12,
    `Scorecard: ${path.relative(repoRoot, path.join(scorecardDir, "REPORT.md"))}\n` +
    `QPU clock coverage: ${path.relative(repoRoot, path.join(scorecardDir, "qpu/archived_method_coverage.csv"))}\n` +
    `Simulator availability: ${path.relative(repoRoot, path.join(scorecardDir, "simulator/availability_matrix.csv"))}\n` +
    `Simulator configuration metrics: ${path.relative(repoRoot, path.join(scorecardDir, "simulator/per_configuration_metrics.csv"))}\n` +
    `Maestro pilot: ${path.relative(repoRoot, path.join(scorecardDir, "simulator/maestro_pilot_summary.csv"))}\n` +
    `Package README: ${path.relative(repoRoot, path.join(repoRoot, "work/repo_finalization/report_package/README.md"))}\n` +
    `Report authority: ${path.relative(repoRoot, path.join(repoRoot, "benchmark_v1/execution/manifests/report_finalization.json"))}`);
  addTable(slide, [
    ["Domain", "Clock or outcome", "Interpretation"],
    ["QPU target", "Archived observed service/execution", "Evaluation target, not schedule duration"],
    ["QPU analytical", "Scheduled single-shot / shot-scaled", "Proxy; provider overhead excluded"],
    ["Simulator", "First / warm execution; quality gate", "Keep engine, config, and clock separate"],
    ["cuTensorNet", "Selected-plan estimate", "Compare only with contraction under same plan"],
    ["Maestro", "Process-isolated reported execution", "Fresh process, terminal pilot failure"],
    ["Package status", "Local review stage", "Validation is not full replication; clean-clone pass not claimed"],
    ["Public release", "Rights decision open", "No public-release approval from this deck"],
  ], { left: 70, top: 145, width: 1140, height: 460 }, { fontSize: 17, headerFontSize: 18, columnWidths: [220, 380, 540] });
  addBody(slide, "Unavailable rows, timeouts, adapter errors, and quality failures stay in coverage denominators.", {
    left: 77, top: 620, width: 1110, height: 35,
  }, { fontSize: 19, bold: true, color: C.teal });
}

// 13. Full 24-variant coverage detail, represented legibly as native tables.
{
  const slide = newSlide("Appendix: coverage của 24 QPU variants", 13,
    `Full coverage figure (companion): ${path.relative(repoRoot, path.join(figureDir, figures.coverage))}\n` +
    `Figure SHA-256: d8dc1661d0bd415eda6c33f8bcad935eb4fed424b832504a138b89225c20f365\n` +
    `Readable native table source: ${path.relative(repoRoot, path.join(scorecardDir, "qpu/archived_method_coverage.csv"))}\n` +
    `Source table SHA-256: ee408a85b5fef378103a2ada6709c791f9e66a8db192e15445bdd6e9803aa915\n` +
    `Scorecard manifest: ${path.relative(repoRoot, path.join(scorecardDir, "manifest.json"))}\n` +
    "Exact row set: all 24 method_id rows in archived_method_coverage.csv; assigned denominator 8,767 per row. Predicted counts and coverage percentages reproduce the source table. Method-output clock labels are abbreviated in the table: service=archived observed service; cal. service=outer-train calibrated observed service; sched.=scheduled duration; T_eff=one-circuit effective cost. Coverage only, not accuracy; method clocks and fidelity differ, so rows are not ranked.",
    { noFooter: true });
  const leftRows = [
    ["Variant", "Pred / 8,767", "Cover", "Output clock"],
    ["MaLi graph V3", "8,763", "99.95%", "Observed service"],
    ["MaLi graph V3-large", "8,766", "99.99%", "Observed service"],
    ["Unified polynomial V3", "8,767", "100.00%", "Observed service"],
    ["QCRE CP raw", "8,766", "99.99%", "Scheduled CP"],
    ["QCRE CP affine", "8,766", "99.99%", "Cal. service"],
    ["QCRE CP log-affine", "8,766", "99.99%", "Cal. service"],
    ["QCRE shot raw", "8,766", "99.99%", "Shot schedule"],
    ["QCRE shot affine", "8,766", "99.99%", "Cal. service"],
    ["QCRE shot log-affine", "8,766", "99.99%", "Cal. service"],
    ["Qiskit duration raw", "8,766", "99.99%", "Scheduled 1-shot"],
    ["Qiskit duration affine", "8,766", "99.99%", "Cal. service"],
    ["Qiskit duration log-affine", "8,766", "99.99%", "Cal. service"],
  ];
  const rightRows = [
    ["Variant", "Pred / 8,767", "Cover", "Output clock"],
    ["Qiskit shot raw", "8,766", "99.99%", "Shot schedule"],
    ["Qiskit shot affine", "8,766", "99.99%", "Cal. service"],
    ["Qiskit shot log-affine", "8,766", "99.99%", "Cal. service"],
    ["Hyb 1-circuit raw", "8,420", "96.04%", "T_eff"],
    ["Hyb 1-circuit affine", "8,420", "96.04%", "Cal. service"],
    ["Hyb 1-circuit log-affine", "8,420", "96.04%", "Cal. service"],
    ["Hyb shot-linear raw", "8,420", "96.04%", "Shot-linear cost"],
    ["Hyb shot-linear affine", "8,420", "96.04%", "Cal. service"],
    ["Hyb shot-linear log-affine", "8,420", "96.04%", "Cal. service"],
    ["Scholten nominal raw", "5,769", "65.80%", "CLOPSv-like"],
    ["Scholten affine", "5,769", "65.80%", "Cal. service"],
    ["Scholten log-affine", "5,769", "65.80%", "Cal. service"],
  ];
  addTable(slide, leftRows, { left: 48, top: 126, width: 574, height: 550 }, { fontSize: 15, headerFontSize: 15, columnWidths: [226, 110, 82, 156] });
  addTable(slide, rightRows, { left: 658, top: 126, width: 574, height: 550 }, { fontSize: 15, headerFontSize: 15, columnWidths: [226, 110, 82, 156] });
}

if (presentation.slides.count !== 13) {
  throw new Error(`Expected 13 slides, found ${presentation.slides.count}`);
}

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
const candidatePath = path.join(buildDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const result = await finalizePresentation({
  explicitTotalSlideCount: 13,
  requiredNativeTableOwnerSlides: [2, 3, 4, 7, 10, 11, 12, 13],
  requiredNativeChartOwnerSlides: [6],
  materializeLiteralChartWorkbooks: true,
  workspaceDir,
  candidatePath,
  finalPath: finalPptx,
  pythonExecutable: process.env.RUNTIME_PYTHON,
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-heading-fit",
    ...[2, 3, 4, 7, 10, 11, 12, 13].flatMap((n) => ["--require-native-table-slide", String(n)]),
  ],
  fontPolicy: { basis: "design", families: [fontFamily] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, "validation.json"),
});
console.log(JSON.stringify({ finalPptx, result }, null, 2));
