
"use strict";

// Находим элементы на странице
const fileInput = document.getElementById("texts-file-input");
const fileNameSpan = document.getElementById("texts-file-name");
const runBtn = document.getElementById("run-analysis-btn");
const resultsBlock = document.getElementById("results-block");
const resultsTableBody = document.querySelector("#results-table tbody");
const downloadBtn = document.getElementById("download-btn");

let lastCsvBlobUrl = null;
let uploadedSrcById = null;   // ID -> src из исходного файла (ID,text,src)
let srcStatsGlobal = null;    // статистика по src для графика

// ========== 1) Обновление названия файла при выборе ==========
if (fileInput && fileNameSpan) {
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      fileNameSpan.textContent = fileInput.files[0].name;
    } else {
      fileNameSpan.textContent = "Файл не выбран";
    }
  });
}

// ========== 2) Обработка кнопки "Запустить анализ" ==========
if (runBtn) {
  runBtn.addEventListener("click", async () => {
    if (!fileInput) {
      console.error("fileInput не найден в DOM");
      return;
    }

    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      alert("Сначала выберите CSV-файл.");
      return;
    }

    // 0) Читаем ИСХОДНЫЙ файл (ID,text,src), строим карту ID -> src
    let originalCsvText = "";               // <-- объявили здесь, снаружи try
    try {
      originalCsvText = await readUploadedFileAsText(file);   // <-- просто присваиваем
      uploadedSrcById = buildSrcMapFromCsv(originalCsvText);
      console.log("📌 uploadedSrcById:", uploadedSrcById);
    } catch (e) {
      console.warn("Не удалось прочитать исходный файл для src:", e);
      uploadedSrcById = null;
    }

    const formData = new FormData();
    formData.append("file", file, file.name);

    // Меняем текст и блокируем кнопку
    runBtn.disabled = true;
    runBtn.textContent = "Анализируем...";

    try {
      const response = await fetch("http://127.0.0.1:8000/api/predict-file", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errText = await response.text();
        console.error("Ошибка от сервера:", errText);
        alert("Ошибка сервера: " + errText);
        return;
      }

      // Читаем CSV как текст (ОТВЕТ сервера: ID,label)
      const csvText = await response.text();
      console.log("🔍 CSV от сервера (первые 200 символов):", csvText.slice(0, 200));

      // --- Автоматическое скачивание файла для пользователя (ID,label) ---
      {
        const blob = new Blob([csvText], { type: "text/csv" });
        const url = URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = "predicted_labels.csv";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }

      // --- Парсим CSV (ожидаем заголовки ID,label) и заполняем таблицу ---
      if (resultsTableBody) {
        const lines = csvText.trim().split("\n");
        if (lines.length > 1) {
          const header = lines[0].split(",");
          const idIndex = header.findIndex(
            (h) => h.trim().toLowerCase() === "id"
          );
          const labelIndex = header.findIndex(
            (h) => h.trim().toLowerCase() === "label"
          );

          if (idIndex === -1 || labelIndex === -1) {
            console.warn("В CSV не найдены колонки ID и label в заголовке");
          }

          const dataRows = lines.slice(1); // без заголовка
          resultsTableBody.innerHTML = "";

          dataRows.forEach((line) => {
            if (!line.trim()) return;
            const cols = line.split(",");
            const id = cols[idIndex] ?? "";
            const label = cols[labelIndex] ?? "";

            const tr = document.createElement("tr");
            tr.innerHTML = `
              <td>${id}</td>
              <td>${label}</td>
            `;
            resultsTableBody.appendChild(tr);
          });
        }
      }

      // --- 3) Обновление графиков распределения ---
      updateClassDistributionChartFromCsv(csvText);                 // общая гистограмма

      if (originalCsvText) {                                        // только если смогли прочитать исходный файл
        updateSourceDistributionFromCsv(originalCsvText, csvText);  // по источникам (join по ID)
      }

      // --- Кнопка "Скачать" (если нужна отдельная) ---
      if (downloadBtn) {
        if (lastCsvBlobUrl) {
          URL.revokeObjectURL(lastCsvBlobUrl);
        }
        const blob = new Blob([csvText], { type: "text/csv" });
        lastCsvBlobUrl = URL.createObjectURL(blob);

        downloadBtn.onclick = () => {
          const a = document.createElement("a");
          a.href = lastCsvBlobUrl;
          a.download = "predicted_labels.csv";
          document.body.appendChild(a);
          a.click();
          a.remove();
        };
      }

      // Показываем блок с результатами (таблица)
      if (resultsBlock) {
        resultsBlock.classList.remove("hidden");
      }
    } catch (e) {
      console.error("Ошибка при запросе:", e);
      alert(
        "Не удалось отправить файл. Проверь, что сервер запущен и доступен по http://127.0.0.1:8000"
      );
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = "Запустить анализ";
    }
  });
}


// ========== Функция: общая гистограмма по классам ==========
function updateClassDistributionChartFromCsv(csvText) {
  const lines = csvText.trim().split("\n");
  if (lines.length <= 1) {
    console.warn("CSV пустой или только с заголовком, график не обновим");
    return;
  }

  const header = lines[0].split(",");
  const labelIndex = header.findIndex(
    (h) => h.trim().toLowerCase() === "label"
  );
  if (labelIndex === -1) {
    console.warn("В CSV нет колонки 'label' в заголовке");
    return;
  }

  const dataRows = lines.slice(1);

  const counts = { 0: 0, 1: 0, 2: 0 };
  let total = 0;

  dataRows.forEach((line) => {
    if (!line.trim()) return;
    const cols = line.split(",");
    const rawLabel = cols[labelIndex];
    const label = Number(rawLabel);
    if (label === 0 || label === 1 || label === 2) {
      counts[label] += 1;
      total += 1;
    }
  });

  [0, 1, 2].forEach((label, idx) => {
    const bar = document.getElementById(`chart-bar-${label}`);
    const valueEl = document.getElementById(`chart-count-${label}`);
    if (!bar || !valueEl) return;

    const fill = bar.querySelector(".app-chart-bar__fill");
    const count = counts[label];
    const percent = total > 0 ? (count / total) * 100 : 0;

    // чтобы совсем мелкий класс тоже был заметен
    const heightPercent =
      total > 0 && count > 0 ? Math.max(percent, 8) : 0;

    if (fill) {
      // маленький сдвиг анимации между столбиками
      fill.style.transitionDelay = `${idx * 0.05}s`;
      fill.style.height = `${heightPercent}%`;
    }

    valueEl.textContent =
      total > 0 ? `${count} (${percent.toFixed(1)}%)` : "0";
  });
}

// ========== Функция: считаем распределение по src ==========
function updateSourceDistributionFromCsv(originalCsvText, predCsvText) {
  // ---- 1. Парсим исходный файл: строим ID -> src ----
  const origLines = originalCsvText.trim().split("\n");
  if (origLines.length <= 1) {
    console.warn("Исходный CSV пустой или только с заголовком, по src нечего считать");
    return;
  }

  const origHeader = parseCsvLine(origLines[0].replace(/\r$/, ""));
  const origIdIndex = origHeader.findIndex(
    (h) => h.trim().toLowerCase() === "id"
  );
  const origSrcIndex = origHeader.findIndex(
    (h) => h.trim().toLowerCase() === "src"
  );

  if (origIdIndex === -1 || origSrcIndex === -1) {
    console.warn("В исходном файле нет колонок ID и/или src");
    return;
  }

  const origDataRows = origLines.slice(1);
  const idToSrc = {}; // ID -> src

  origDataRows.forEach((rawLine) => {
    const line = rawLine.trim();
    if (!line) return;
    const cols = parseCsvLine(line.replace(/\r$/, ""));

    const id = (cols[origIdIndex] ?? "").trim();
    const src = (cols[origSrcIndex] ?? "").trim() || "Не указан";

    if (id) {
      idToSrc[id] = src;
    }
  });

  // ---- 2. Парсим файл предсказаний: ID,label ----
  const predLines = predCsvText.trim().split("\n");
  if (predLines.length <= 1) {
    console.warn("CSV с предсказаниями пустой или только с заголовком");
    return;
  }

  const predHeader = parseCsvLine(predLines[0].replace(/\r$/, ""));
  const predIdIndex = predHeader.findIndex(
    (h) => h.trim().toLowerCase() === "id"
  );
  const predLabelIndex = predHeader.findIndex(
    (h) => h.trim().toLowerCase() === "label"
  );

  if (predIdIndex === -1 || predLabelIndex === -1) {
    console.warn("В CSV с предсказаниями нет колонок ID и/или label");
    return;
  }

  const predDataRows = predLines.slice(1);

  // ---- 3. Join по ID: src × label ----
  const statsBySrc = {}; // src -> {0,1,2,total}

  predDataRows.forEach((rawLine) => {
    const line = rawLine.trim();
    if (!line) return;
    const cols = parseCsvLine(line.replace(/\r$/, ""));

    const id = (cols[predIdIndex] ?? "").trim();
    const rawLabel = cols[predLabelIndex];
    const label = Number(rawLabel);

    if (!(label === 0 || label === 1 || label === 2)) return;

    const src = idToSrc[id] || "Не указан";

    if (!statsBySrc[src]) {
      statsBySrc[src] = { 0: 0, 1: 0, 2: 0, total: 0 };
    }

    statsBySrc[src][label] += 1;
    statsBySrc[src].total += 1;
  });

  // ---- 4. Обновляем select и рисуем график ----
  const srcSelect = document.getElementById("src-select");
  if (!srcSelect) return;

  const entries = Object.entries(statsBySrc).sort(
    (a, b) => b[1].total - a[1].total
  );

  srcSelect.innerHTML = "";

  if (entries.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "— нет данных —";
    srcSelect.appendChild(opt);
    renderSourceChart(null, statsBySrc);
    return;
  }

  entries.forEach(([src, stat], idx) => {
    const opt = document.createElement("option");
    opt.value = src;
    opt.textContent = `${src} (${stat.total})`;
    if (idx === 0) opt.selected = true;
    srcSelect.appendChild(opt);
  });

  srcSelect.onchange = () => {
    const value = srcSelect.value || null;
    renderSourceChart(value, statsBySrc);
  };

  const firstSrc = entries[0][0];
  renderSourceChart(firstSrc, statsBySrc);
}



// ========== Функция: рисуем 3 столбика для выбранного src ==========
function renderSourceChart(src, statsBySrc) {
  const bars = [0, 1, 2];

  const countsEls = {
    0: document.getElementById("src-count-0"),
    1: document.getElementById("src-count-1"),
    2: document.getElementById("src-count-2"),
  };

  if (!src || !statsBySrc || !statsBySrc[src]) {
    // Обнуляем график
    bars.forEach((label) => {
      const bar = document.getElementById(`src-bar-${label}`);
      const fill = bar ? bar.querySelector(".app-chart-bar__fill") : null;
      if (fill) fill.style.height = "0%";
      if (countsEls[label]) countsEls[label].textContent = "0";
    });
    return;
  }

  const stat = statsBySrc[src];
  const total = stat.total || 0;

  bars.forEach((label, idx) => {
    const bar = document.getElementById(`src-bar-${label}`);
    const fill = bar ? bar.querySelector(".app-chart-bar__fill") : null;
    const countEl = countsEls[label];

    const count = stat[label] || 0;
    const percent = total > 0 ? (count / total) * 100 : 0;
    const heightPercent =
      total > 0 && count > 0 ? Math.max(percent, 8) : 0;

    if (fill) {
      fill.style.transitionDelay = `${idx * 0.05}s`;
      fill.style.height = `${heightPercent}%`;
    }

    if (countEl) {
      countEl.textContent =
        total > 0 ? `${count} (${percent.toFixed(1)}%)` : "0";
    }
  });
}



// ========== Вспомогательные функции для чтения исходного файла ==========
function readUploadedFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsText(file, "utf-8");
  });
}

function buildSrcMapFromCsv(csvText) {
  const map = {};
  const lines = csvText.trim().split("\n");
  if (lines.length <= 1) return map;

  // заголовок: парсим с учётом кавычек
  const header = parseCsvLine(lines[0].replace(/\r$/, ""));
  const idIndex = header.findIndex(
    (h) => h.trim().toLowerCase() === "id"
  );
  const srcIndex = header.findIndex(
    (h) => h.trim().toLowerCase() === "src"
  );

  if (idIndex === -1 || srcIndex === -1) {
    console.warn("В исходном файле нет колонок ID и/или src");
    return map;
  }

  const dataRows = lines.slice(1);

  dataRows.forEach((rawLine) => {
    const line = rawLine.trim();
    if (!line) return;

    const cols = parseCsvLine(line.replace(/\r$/, "")); // <-- вместо split(",")

    const id = (cols[idIndex] ?? "").trim();
    const src = (cols[srcIndex] ?? "").trim();

    if (id) {
      map[id] = src || "Не указан";
    }
  });

  return map;
}



function parseCsvLine(line) {
  const result = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];

    if (ch === '"') {
      // удвоенные кавычки внутри поля -> одна кавычка
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++; // пропускаем вторую кавычку
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      // запятая вне кавычек — разделитель поля
      result.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  result.push(current);

  return result;
}


// Аккуратный разбор строки CSV с кавычками
function parseCsvLine(line) {
  const result = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];

    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      result.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  result.push(current);
  return result;
}

function readUploadedFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsText(file, "utf-8");
  });
}

// Парсим CSV формата ID,label -> возвращаем Map: id -> label (число)
function buildIdLabelMap(csvText) {
  const map = new Map();

  // 1) Разбили текст на строки
  const lines = csvText.trim().split(/\r?\n/);
  if (lines.length <= 1) return map;

  // 2) Вот тут можно посмотреть «сырую» первую строку
  console.log("RAW header line:", JSON.stringify(lines[0]));

  const rawHeaderLine = lines[0];
  const header = parseCsvLine(rawHeaderLine);

  console.log("header parsed:", header);

  const idIndex = header.findIndex(
    (h) => normalizeHeaderName(h) === "id"
  );
  const labelIndex = header.findIndex(
    (h) => normalizeHeaderName(h) === "label"
  );

  if (idIndex === -1 || labelIndex === -1) {
    console.warn("В файле нет колонок ID и/или label", header);
    return map;
  }

  const dataRows = lines.slice(1);

  dataRows.forEach((rawLine) => {
    const line = rawLine.trim();
    if (!line) return;

    const cols = parseCsvLine(line);
    const id = (cols[idIndex] ?? "").trim();
    const labelRaw = (cols[labelIndex] ?? "").trim();

    if (!id) return;
    const labelNum = Number(labelRaw);
    if (!(labelNum === 0 || labelNum === 1 || labelNum === 2)) return;

    map.set(id, labelNum);
  });

  return map;
}


function computeMetricsFromMaps(trueMap, predMap) {
  const labels = [0, 1, 2];
  const labelIndex = { 0: 0, 1: 1, 2: 2 };

  // 3x3 confusion matrix: [true][pred]
  const conf = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];

  let total = 0;
  let correct = 0;

  // Берём только IDs, которые есть в обоих файлах
  trueMap.forEach((trueLabel, id) => {
    if (!predMap.has(id)) return;
    const predLabel = predMap.get(id);

    if (!(labels.includes(trueLabel) && labels.includes(predLabel))) return;

    const i = labelIndex[trueLabel];
    const j = labelIndex[predLabel];

    conf[i][j] += 1;
    total += 1;
    if (trueLabel === predLabel) correct += 1;
  });

  if (total === 0) {
    return {
      accuracy: 0,
      macroPrecision: 0,
      macroRecall: 0,
      macroF1: 0,
      confusion: conf,
      total: 0,
    };
  }

  // precision, recall, F1 по классам
  const precisions = [];
  const recalls = [];
  const f1s = [];

  labels.forEach((label) => {
    const idx = labelIndex[label];
    const tp = conf[idx][idx];

    let fp = 0;
    let fn = 0;

    // FP: предсказали label, но истинный другой
    labels.forEach((l2) => {
      const j2 = labelIndex[label];
      const i2 = labelIndex[l2];
      if (l2 !== label) fp += conf[i2][j2];
    });

    // FN: истинный label, но предсказали другой
    labels.forEach((l2) => {
      const i2 = labelIndex[label];
      const j2 = labelIndex[l2];
      if (l2 !== label) fn += conf[i2][j2];
    });

    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 =
      prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;

    precisions.push(prec);
    recalls.push(rec);
    f1s.push(f1);
  });

  const macroPrecision =
    precisions.reduce((s, x) => s + x, 0) / precisions.length;
  const macroRecall =
    recalls.reduce((s, x) => s + x, 0) / recalls.length;
  const macroF1 =
    f1s.reduce((s, x) => s + x, 0) / f1s.length;
  const accuracy = correct / total;

  return {
    accuracy,
    macroPrecision,
    macroRecall,
    macroF1,
    confusion: conf,
    total,
  };
}

function renderMetricsToDom(metrics) {
  const macroF1El = document.getElementById("macro-f1-value");
  const accEl = document.getElementById("accuracy-value");
  const precEl = document.getElementById("precision-value");
  const recEl = document.getElementById("recall-value");

  if (macroF1El) {
    macroF1El.textContent = metrics.macroF1.toFixed(3);
  }
  if (accEl) {
    accEl.textContent = `Accuracy: ${metrics.accuracy.toFixed(3)}`;
  }
  if (precEl) {
    precEl.textContent = `Precision (macro): ${metrics.macroPrecision.toFixed(3)}`;
  }
  if (recEl) {
    recEl.textContent = `Recall (macro): ${metrics.macroRecall.toFixed(3)}`;
  }
}

function renderConfusionToDom(conf) {
  const tbody = document.querySelector("#confusion-table tbody");
  if (!tbody) return;

  const labels = [0, 1, 2];
  const rowsHtml = labels
    .map((trueLabel, i) => {
      const row = conf[i];
      return `
        <tr>
          <th>True ${trueLabel}</th>
          <td>${row[0]}</td>
          <td>${row[1]}</td>
          <td>${row[2]}</td>
        </tr>
      `;
    })
    .join("");

  tbody.innerHTML = rowsHtml;
}

// ===== ЭЛЕМЕНТЫ БЛОКА МЕТРИК (macro-F1) =====
// ===== ЭЛЕМЕНТЫ БЛОКА МЕТРИК (macro-F1) =====
const predFileInput = document.getElementById("pred-file-input");
const predFileNameSpan = document.getElementById("pred-file-name");
const valFileInput = document.getElementById("val-file-input");
const valFileNameSpan = document.getElementById("val-file-name");
const calcMacroBtn = document.getElementById("calc-macro-btn");
const metricsResultsBlock = document.getElementById("metrics-results");

// Обновление названия файла с предсказаниями
if (predFileInput && predFileNameSpan) {
  predFileInput.addEventListener("change", () => {
    if (predFileInput.files && predFileInput.files.length > 0) {
      predFileNameSpan.textContent = predFileInput.files[0].name;
    } else {
      predFileNameSpan.textContent = "Файл не выбран";
    }
  });
}

// Обновление названия файла с истинными метками
if (valFileInput && valFileNameSpan) {
  valFileInput.addEventListener("change", () => {
    if (valFileInput.files && valFileInput.files.length > 0) {
      valFileNameSpan.textContent = valFileInput.files[0].name;
    } else {
      valFileNameSpan.textContent = "Файл не выбран";
    }
  });
}

// Обработчик кнопки "Посчитать macro-F1"
if (calcMacroBtn) {
  calcMacroBtn.addEventListener("click", async () => {
    const predFile = predFileInput && predFileInput.files[0];
    const valFile = valFileInput && valFileInput.files[0];

    if (!predFile || !valFile) {
      alert("Пожалуйста, выберите оба файла: с предсказаниями и с истинными метками.");
      return;
    }

    try {
      const [predCsvText, valCsvText] = await Promise.all([
        readUploadedFileAsText(predFile),
        readUploadedFileAsText(valFile),
      ]);

      const predMap = buildIdLabelMap(predCsvText);
      const trueMap = buildIdLabelMap(valCsvText);

      if (predMap.size === 0 || trueMap.size === 0) {
        alert("Не удалось прочитать метки из одного из файлов. Проверьте формат ID,label.");
        return;
      }

      const metrics = computeMetricsFromMaps(trueMap, predMap);
      renderMetricsToDom(metrics);
      renderConfusionToDom(metrics.confusion);

      if (metricsResultsBlock) {
        metricsResultsBlock.classList.remove("hidden");
      }
    } catch (e) {
      console.error("Ошибка при расчёте метрик:", e);
      alert("Не удалось прочитать файлы. Проверьте формат CSV и попробуйте ещё раз.");
    }
  });
}

function normalizeHeaderName(h) {
  if (!h) return "";
  let s = h.replace(/^\uFEFF/, "").trim(); // убираем BOM в начале, если есть
  if (s.startsWith('"') && s.endsWith('"')) {
    s = s.slice(1, -1); // убираем внешние кавычки
  }
  return s.trim().toLowerCase();
}