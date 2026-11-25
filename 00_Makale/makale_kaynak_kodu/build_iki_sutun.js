// Kullanım: node build.js tr|en  -> Makale_TR.docx / Makale_EN.docx
const fs = require("fs"), path = require("path");
const D = require("docx");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  BorderStyle, ShadingType, TabStopType, Footer, Header, PageNumber, HeadingLevel, VerticalAlign, SectionType } = D;

const L = process.argv[2];
const C = require(`./content_${L}.js`);
const REFS = require("./refs.js");
const FIG = path.join(__dirname, "..", "..", "04_Sekiller", L);
const RES = path.join(__dirname, "..", "..", "03_Ortak_Karsilastirma_ve_Hibrit", "sonuclar");
const FONT = "Times New Roman";
const MG = 1077; // 1,9 cm kenar
const TW = 11906 - 2 * MG; // tam genişlik
const GAP = 400; const CW = Math.floor((TW - GAP) / 2); // sütun genişliği

// ---------- atıf numaralandırma
const order = [];
const scan = s => (s.match(/\[@[a-z0-9]+\]/g) || []).forEach(m => { const k = m.slice(2, -1); if (!REFS[k]) throw "ref yok: " + k; if (!order.includes(k)) order.push(k); });
C.body.forEach(b => { if (b.p) scan(b.p); if (b.cap) scan(b.cap); });
const cite = s => s.replace(/\[@([a-z0-9]+)\]/g, (_, k) => `[${order.indexOf(k) + 1}]`);
const unused = Object.keys(REFS).filter(k => !order.includes(k));
if (unused.length) console.log("Kullanılmayan kaynaklar:", unused);

// ---------- sayı biçimi
const dec = L === "tr" ? "," : ".";
const f = (x, n) => x.toFixed(n).replace(".", dec);
const sci = x => { if (x === 0) return "0"; const e = Math.floor(Math.log10(Math.abs(x))); const m = x / 10 ** e;
  const sup = String(e).split("").map(c => "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"["-0123456789".indexOf(c)]).join("");
  return `${f(m, 1)}×10${sup}`; };
const pf = p => isNaN(p) ? "–" : (p < 0.001 ? (L === "tr" ? "<0,001" : "<0.001") : f(p, 3));

function csv(file) {
  const [h, ...r] = fs.readFileSync(file, "utf8").trim().split("\n").map(l => l.split(","));
  return r.map(row => Object.fromEntries(h.map((k, i) => [k, row[i] === "" ? NaN : (isNaN(+row[i]) ? row[i] : +row[i])])));
}
const ALG = ["GA", "ACO_R", "PSO", "GWO", "GA-ACO_R", "PSO-GWO", "Rastgele Arama"];
const nm = a => a === "Rastgele Arama" ? C.random : a === "Manuel PID" ? C.manual : a;

// ---------- metin yardımcıları
const run = (t, o = {}) => new TextRun({ text: t, font: FONT, size: o.size || 20, bold: o.bold, italics: o.italics });
const para = (t, o = {}) => new Paragraph({
  alignment: o.align || AlignmentType.JUSTIFIED, keepNext: o.keepNext,
  spacing: { line: o.line || 252, before: o.before || 0, after: o.after ?? 40 },
  indent: o.indent ? { firstLine: 284 } : undefined,
  children: Array.isArray(t) ? t : [run(t, o)] });

// "GA." gibi kalın başlangıç
function bodyPara(t) {
  const m = t.match(/^((?:GA|ACO_R|PSO|GWO|Hibrit [12]: [^.]+?|Hybrid [12]: [^.]+?)\.)\s(.*)$/s);
  if (m) return para([run(m[1] + " ", { bold: true }), run(m[2])], { indent: true });
  const m2 = t.match(/^(Sınırlılıklar:|Limitations:)(.*)$/s);
  if (m2) return para([run(m2[1], { bold: true }), run(m2[2])]);
  return para(t, { indent: true });
}

function pngSize(file) { const b = fs.readFileSync(file); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
function figure(b) {
  const file = path.join(FIG, b.fig); const [w, h] = pngSize(file);
  const wpx = Math.min(b.w, 17) / 2.54 * 96; const hpx = wpx * h / w;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: Math.round(wpx), height: Math.round(hpx) } })] }),
    para(cite(b.cap), { size: 17, line: 228, after: 120, align: AlignmentType.JUSTIFIED }),
  ];
}

const bd = { style: BorderStyle.SINGLE, size: 4, color: "000000" }, nb = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function cell(t, w, o = {}) {
  return new TableCell({ width: { size: w, type: WidthType.DXA }, columnSpan: o.span, verticalAlign: VerticalAlign.CENTER,
    borders: { top: o.top ? bd : nb, bottom: o.bottom ? bd : nb, left: nb, right: nb },
    shading: o.shade ? { fill: "EEF3FB", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 30, bottom: 30, left: 60, right: 60 },
    children: [new Paragraph({ alignment: o.left ? AlignmentType.LEFT : AlignmentType.CENTER, spacing: { line: 240 }, keepNext: !o.last, keepLines: true,
      children: [run(t, { size: o.size, bold: o.bold })] })] });
}
function table(key) {
  const T = C.tables[key]; const sz = T.small ? 15 : 16;
  const tot = T.widths.reduce((a, b) => a + b, 0); const W = T.widths.map(x => Math.floor(x * TW / tot));
  W[0] += TW - W.reduce((a, b) => a + b, 0);
  let rows = T.rows, boldCells = new Set();
  if (key === "t2") {
    const K = Object.fromEntries(csv(path.join(RES, "kalkis", "ozet_egitim.csv")).map(r => [r.yontem, r]));
    const I = Object.fromEntries(csv(path.join(RES, "inis", "ozet_egitim.csv")).map(r => [r.yontem, r]));
    const best = O => ALG.slice(0, 6).reduce((a, b) => O[a].mean < O[b].mean ? a : b);
    const bk = best(K), bi = best(I);
    const part = (r, isR) => [f(r.mean, 6), sci(r.std), isR ? "– (0/10)" : `${Math.round(r["eval_%1_ort"])} (${r["basari_%1"]})`,
      isR ? "–" : f(r.ort_sira, 1), pf(r.p_eniyi_ile)];
    rows = ALG.map((a, i) => { if (a === bk) boldCells.add(`${i},1`); if (a === bi) boldCells.add(`${i},6`);
      return [nm(a), ...part(K[a], a === "Rastgele Arama"), ...part(I[a], a === "Rastgele Arama")]; });
  }
  if (key === "t3") {
    const K = Object.fromEntries(csv(path.join(RES, "kalkis", "test_sonuclari.csv")).map(r => [r.yontem, r]));
    const I = Object.fromEntries(csv(path.join(RES, "inis", "test_sonuclari.csv")).map(r => [r.yontem, r]));
    rows = ["Manuel PID", ...ALG].map(a => [nm(a), f(K[a].J_test, 4), f(K[a].IAE_ort, 1), f(K[a].TV_ort, 1), `${K[a].maxQ_ihlal}/200`,
      f(I[a].J_test, 4), f(I[a].IAE_ort, 1), f(I[a].v_temas_ort, 2), f(I[a].v_temas_maks, 2), f(I[a].yakit_ort_ton, 1)]);
  }
  const out = [];
  if (T.head2) { let c = 0; out.push(new TableRow({ tableHeader: true, children: T.head2.map(([t, s]) => {
    const w = W.slice(c, c + s).reduce((a, b) => a + b, 0); c += s;
    return cell(t, w, { span: s > 1 ? s : undefined, top: true, bold: true, size: sz, shade: true }); }) })); }
  out.push(new TableRow({ tableHeader: true, children: T.head.map((t, i) => cell(t, W[i], { top: !T.head2, bottom: true, bold: true, size: sz, shade: true, left: i === 0 && !T.head2 })) }));
  rows.forEach((r, ri) => out.push(new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, W[i],
    { size: sz, left: i === 0 || (key === "t1"), bottom: ri === rows.length - 1, last: ri === rows.length - 1, bold: boldCells.has(`${ri},${i}`) })) })));
  return [para(T.cap, { size: 17, line: 228, after: 60, before: 80, keepNext: true, align: AlignmentType.LEFT }),
    new Table({ width: { size: TW, type: WidthType.DXA }, columnWidths: W, rows: out }),
    para("", { after: 60, line: 120 })];
}

function splitEq(s) {
  const MAX = 44; let parts = s.split(/,\s{2,}/);
  const out = [];
  for (let p of parts) {
    while (p.length > MAX) {
      let cut = -1, i = -1;
      while ((i = p.indexOf(" + ", i + 1)) !== -1 && i < MAX) cut = i;
      if (cut <= 0) break;
      out.push(p.slice(0, cut)); p = p.slice(cut + 1);
    }
    out.push(p);
  }
  return out;
}
function equation(b) {
  const ls = splitEq(b.eq.replace(/^\S+:\s+/, m => m.trim() + " "));
  return ls.map((l, k) => new Paragraph({ tabStops: [{ type: TabStopType.CENTER, position: CW / 2 }, { type: TabStopType.RIGHT, position: CW }],
    spacing: { line: 252, before: k === 0 ? 60 : 0, after: k === ls.length - 1 ? 60 : 0 }, keepNext: k < ls.length - 1, keepLines: true,
    children: [run("\t" + l, { size: 18 }), run(k === ls.length - 1 ? `\t(${b.n})` : "", { size: 18 })] }));
}

// ---------- belge
const hdr = new Header({ children: [new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "808080", space: 2 } },
  tabStops: [{ type: TabStopType.RIGHT, position: TW }], children: [
  new TextRun({ text: L === "tr" ? "Optimizasyon Dersi Bildirisi · Kırıkkale Üniversitesi" : "Optimization Course Paper · Kırıkkale University", font: FONT, size: 16, italics: true, color: "555555" }),
  new TextRun({ text: "\t" + (L === "tr" ? "Aralık 2026" : "December 2026"), font: FONT, size: 16, italics: true, color: "555555" })] })] });
const ftr = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] });
const page = { size: { width: 11906, height: 16838 }, margin: { top: 1077, bottom: 1077, left: MG, right: MG, header: 600, footer: 500 } };
const sections = []; let cur = [], curCols = 1;
const flush = (cols) => { if (cur.length) sections.push({ properties: { page, ...(sections.length ? { type: SectionType.CONTINUOUS } : {}),
  column: cols === 2 ? { count: 2, space: GAP, equalWidth: true } : { count: 1 } }, headers: { default: hdr }, footers: { default: ftr }, children: cur }); cur = []; };
const go = (cols) => { if (cols !== curCols) { flush(curCols); curCols = cols; } };
const rule = () => new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "1F4E8C", space: 4 } }, spacing: { after: 160 }, children: [] });

// Başlık bloğu (tek sütun)
cur.push(para(C.title, { align: AlignmentType.CENTER, size: 30, bold: true, line: 276, after: 160 }));
cur.push(para(C.authors, { align: AlignmentType.CENTER, size: 22, bold: true, line: 276, after: 40 }));
cur.push(para(C.affil, { align: AlignmentType.CENTER, size: 18, italics: true, line: 240, after: 30 }));
cur.push(para(C.email, { align: AlignmentType.CENTER, size: 18, line: 240, after: 30 }));
cur.push(para(C.note, { align: AlignmentType.CENTER, size: 16, line: 240, after: 120 }));
cur.push(rule());
cur.push(para([run(C.absTitle + ". ", { bold: true, size: 19 }), run(C.abstract, { size: 19 })], { line: 240, after: 80 }));
cur.push(para([run(C.kwTitle + ": ", { bold: true, size: 19 }), run(C.keywords, { size: 19, italics: true })], { after: 80, line: 240 }));
cur.push(rule());
go(2);
const wide = b => (b.fig) || (b.table);
for (const b of C.body) {
  if (wide(b)) { go(1); cur.push(...(b.fig ? figure(b) : table(b.table))); go(2); continue; }
  if (b.h1 || b.h1u) cur.push(new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, spacing: { before: 200, after: 80 },
    children: [run((b.h1 || b.h1u).toLocaleUpperCase(L === "tr" ? "tr-TR" : "en-US"), { bold: true, size: 20 })] }));
  else if (b.h2) cur.push(new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, spacing: { before: 120, after: 60 },
    children: [run(b.h2, { bold: true, italics: true, size: 20 })] }));
  else if (b.p) cur.push(bodyPara(cite(b.p)));
  else if (b.eq) cur.push(...equation(b));
}
cur.push(new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, spacing: { before: 200, after: 80 }, children: [run(C.refTitle.toLocaleUpperCase(L === "tr" ? "tr-TR" : "en-US"), { bold: true, size: 20 })] }));
order.forEach((k, i) => cur.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { line: 228, after: 20 },
  indent: { left: 400, hanging: 400 }, children: [run(`[${i + 1}]\t${REFS[k]}`, { size: 16 })], tabStops: [{ type: TabStopType.LEFT, position: 400 }] })));
flush(curCols);

const doc = new Document({
  creator: "İrem Sert, Melih Demir", title: C.title,
  styles: { default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 20, bold: true, color: "000000" }, paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 20, bold: true, italics: true, color: "000000" }, paragraph: { spacing: { before: 120, after: 60 }, keepNext: true, outlineLevel: 1 } }] },
  sections,
});
const out = `Makale_${L.toUpperCase()}.docx`;
Packer.toBuffer(doc).then(b => { fs.writeFileSync(out, b); console.log(out, "atıf sayısı:", order.length); });
