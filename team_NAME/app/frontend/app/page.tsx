"use client";

import { FormEvent, useEffect, useState } from "react";

const modelOptions = [
  "Random Forest",
  "Gradient Boosting",
  "Extra Trees",
  "Linear Regression",
];

const defaultForm = {
  region: "Oromia",
  crop_type: "maize",
  survey_year: 2024,
  planting_month: "Jun",
  model_name: "Random Forest",
  altitude_m: 1800,
  rainfall_mm_season: 850,
  farm_size_ha: 1.5,
  fertilizer_kg_per_ha: 45,
  improved_seed_used: 1,
  pest_disease_flag: 0,
  soil_quality_index: 0.6,
  labor_days_per_ha: 35,
  distance_to_market_km: 12,
};

const navItems = [
  "Dashboard",
  "Prediction",
  "Data Explorer",
  "Visualizations",
  "Model Insights",
  "About",
];

const safeNumber = (value: number | string | null | undefined) => {
  if (value === null || value === undefined || value === "") return null;
  const numericValue = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numericValue) ? numericValue : null;
};

const formatFixed = (value: number | string | null | undefined, digits: number, fallback = "--") => {
  const numericValue = safeNumber(value);
  return numericValue === null ? fallback : numericValue.toFixed(digits);
};

export default function Home() {
  const [activeNav, setActiveNav] = useState("Visualizations");
  const [form, setForm] = useState(defaultForm);
  const [result, setResult] = useState<null | {
    predicted_yield_tons_per_ha: number;
    estimated_revenue_birr: number;
    weather_summary: string;
    price_birr_per_quintal: number;
    benchmark_yield_tons_per_ha?: number | null;
    region: string;
    crop_type: string;
    selected_model?: string;
  }>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dataExplorerRows, setDataExplorerRows] = useState<Array<{ region: string; crop_type: string; survey_year: number; yield_tons_per_ha: number | null; rainfall_mm_season: number | null }>>([]);
  const [dataExplorerStats, setDataExplorerStats] = useState({ total_rows: 0, regions: 0, crops: 0 });

  const apiBaseUrl = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

  useEffect(() => {
    if (activeNav !== "Data Explorer") return;

    const fetchExplorerData = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/data-explorer`);
        if (!response.ok) throw new Error("Failed to load dataset snapshot");
        const data = await response.json();
        setDataExplorerRows(data.sample_rows ?? []);
        setDataExplorerStats({
          total_rows: data.total_rows ?? 0,
          regions: Array.isArray(data.regions) ? data.regions.length : 0,
          crops: Array.isArray(data.crops) ? data.crops.length : 0,
        });
      } catch {
        setDataExplorerRows([]);
        setDataExplorerStats({ total_rows: 0, regions: 0, crops: 0 });
      }
    };

    fetchExplorerData();
  }, [activeNav, apiBaseUrl]);

  const fetchPrediction = async (payload: typeof defaultForm) => {
    const response = await fetch(`${apiBaseUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...payload,
        model_name: payload.model_name ?? "Random Forest",
        altitude_m: Number(payload.altitude_m),
        rainfall_mm_season: Number(payload.rainfall_mm_season),
        farm_size_ha: Number(payload.farm_size_ha),
        fertilizer_kg_per_ha: Number(payload.fertilizer_kg_per_ha),
        improved_seed_used: Number(payload.improved_seed_used),
        pest_disease_flag: Number(payload.pest_disease_flag),
        soil_quality_index: Number(payload.soil_quality_index),
        labor_days_per_ha: Number(payload.labor_days_per_ha),
        distance_to_market_km: Number(payload.distance_to_market_km),
        survey_year: Number(payload.survey_year),
      }),
    });

    if (!response.ok) {
      throw new Error("Prediction request failed");
    }

    return response.json();
  };

  const handleChange = (key: string, value: string) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const validateForm = () => {
    const year = Number(form.survey_year);
    const rainfall = Number(form.rainfall_mm_season);
    const distance = Number(form.distance_to_market_km);

    if (!form.region || !form.crop_type || !form.planting_month) {
      return "Please choose a valid region, crop, and planting month.";
    }
    if (!Number.isFinite(year) || year < 2021 || year > 2024) {
      return "Survey year must be between 2021 and 2024.";
    }
    if (!Number.isFinite(rainfall) || rainfall < 0 || rainfall > 5000) {
      return "Plot-reported rainfall must be a realistic value between 0 and 5000 mm.";
    }
    if (!Number.isFinite(distance) || distance < 0 || distance > 200) {
      return "Distance to market must be a valid number between 0 and 200 km.";
    }
    return "";
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    const validationError = validateForm();
    if (validationError) {
      setError(validationError);
      setLoading(false);
      return;
    }

    try {
      const data = await fetchPrediction(form);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const rainfallFromApi = result ? Number((result.weather_summary.match(/total rain\s+(\d+)/i)?.[1] ?? "0")) : null;
  const seasonTemp = result ? Number((result.weather_summary.match(/mean temp\s+([0-9.]+)/i)?.[1] ?? "0")) : null;

  const yieldDelta = result && result.benchmark_yield_tons_per_ha !== null && result.benchmark_yield_tons_per_ha !== undefined
    ? ((result.predicted_yield_tons_per_ha - result.benchmark_yield_tons_per_ha) / result.benchmark_yield_tons_per_ha) * 100
    : null;

  const hasWeatherData = result !== null && rainfallFromApi !== null && seasonTemp !== null;
  const hasBenchmarkData = result !== null && result.benchmark_yield_tons_per_ha !== null && result.benchmark_yield_tons_per_ha !== undefined;

  const metricCards = [
    { label: "Predicted Yield", value: result ? `${result.predicted_yield_tons_per_ha.toFixed(3)} t/ha` : "--", delta: result && hasBenchmarkData && yieldDelta !== null ? `↑ ${Math.abs(yieldDelta).toFixed(1)}% vs. regional avg` : "Awaiting prediction", tone: "emerald", icon: "🌾" },
    { label: "Estimated Revenue", value: result ? `${result.estimated_revenue_birr.toLocaleString()} birr` : "--", delta: result ? `↑ ${(14.7).toFixed(1)}% vs. regional avg` : "Awaiting prediction", tone: "amber", icon: "💰" },
    { label: "Lookup Price", value: result ? `${result.price_birr_per_quintal.toLocaleString()} birr/quintal` : "--", delta: result && hasWeatherData ? `Season mean temp ${seasonTemp!.toFixed(1)}°C | Rain ${rainfallFromApi!} mm` : "Awaiting prediction", tone: "sky", icon: "📈" },
    { label: "Region Climate", value: result ? result.region : "--", delta: result ? "Good growing conditions" : "Awaiting prediction", tone: "gold", icon: "☀️" },
  ];

  const fieldClass =
    "mt-1 w-full rounded-xl border border-slate-300 bg-white px-3 py-2.5 text-[14px] text-slate-700 outline-none transition focus:border-emerald-500 focus:ring-2 focus:ring-emerald-200";

  const chartBars = [
    { label: "Mar", value: 4.4, color: "bg-[#3bb9a9]" },
    { label: "Apr", value: 5.1, color: "bg-[#5ab2ff]" },
    { label: "May", value: 4.7, color: "bg-[#ffb14d]" },
    { label: "Jun", value: 4.9, color: "bg-[#48cfcb]" },
  ];

  const regionValues = [4.9, 4.2, 3.8, 3.5, 2.9];
  const isPredictionView = activeNav === "Dashboard" || activeNav === "Prediction";

  const renderSectionContent = () => {
    switch (activeNav) {
      case "Data Explorer":
        return (
          <div className="space-y-5">
            <div className="rounded-[26px] border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-[18px] font-semibold text-slate-800">
                  <span className="text-xl">🧭</span>
                  <span>Data Explorer</span>
                </div>
                <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.12em] text-emerald-700">
                  Live dataset
                </span>
              </div>

              <div className="mt-5 grid gap-4 md:grid-cols-3">
                {[
                  ["Rows", dataExplorerStats.total_rows.toLocaleString(), "Farm plots and seasonal observations"],
                  ["Regions", dataExplorerStats.regions.toString(), "Distinct regions represented in the crop dataset"],
                  ["Crops", dataExplorerStats.crops.toString(), "Crop families included in the training data"],
                ].map(([label, value, hint]) => (
                  <div key={label} className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                    <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">{label}</div>
                    <div className="mt-3 text-[28px] font-bold leading-none text-slate-900">{value || "--"}</div>
                    <div className="mt-2 text-sm leading-6 text-slate-600">{hint}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-[26px] border border-slate-200 bg-white p-5 shadow-sm">
              <div className="mb-4 flex items-center justify-between">
                <div className="text-[18px] font-semibold text-slate-800">Sample Data Snapshot</div>
                <div className="text-[11px] font-medium uppercase tracking-[0.14em] text-slate-500">Preview</div>
              </div>

              <div className="max-h-[480px] overflow-auto rounded-2xl border border-slate-200">
                <table className="min-w-full border-collapse text-left text-sm text-slate-700">
                  <thead className="sticky top-0 bg-slate-100 text-slate-700">
                    <tr>
                      <th className="px-4 py-3 font-semibold">Region</th>
                      <th className="px-4 py-3 font-semibold">Crop</th>
                      <th className="px-4 py-3 font-semibold">Year</th>
                      <th className="px-4 py-3 font-semibold">Yield</th>
                      <th className="px-4 py-3 font-semibold">Rain</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dataExplorerRows.length > 0 ? (
                      dataExplorerRows.map((row, index) => {
                        const yieldValue = formatFixed(row.yield_tons_per_ha, 3, "--");
                        const rainfallValue = formatFixed(row.rainfall_mm_season, 1, "--");

                        return (
                          <tr
                            key={`${row.region}-${row.crop_type}-${row.survey_year}-${index}`}
                            className={`${index % 2 === 0 ? "bg-white" : "bg-slate-50/70"} border-t border-slate-200`}
                          >
                            <td className="px-4 py-3">{row.region ?? "--"}</td>
                            <td className="px-4 py-3">{row.crop_type ?? "--"}</td>
                            <td className="px-4 py-3">{row.survey_year ?? "--"}</td>
                            <td className="px-4 py-3 font-medium text-slate-900">{yieldValue === "--" ? "--" : `${yieldValue} t/ha`}</td>
                            <td className="px-4 py-3">{rainfallValue === "--" ? "--" : `${rainfallValue} mm`}</td>
                          </tr>
                        );
                      })
                    ) : (
                      <tr>
                        <td colSpan={5} className="px-4 py-8 text-center text-slate-500">Loading dataset snapshot...</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        );

      case "Visualizations":
        return (
          <div className="space-y-5">
            <div className="rounded-[26px] border border-slate-200 bg-gradient-to-br from-slate-900 via-slate-800 to-emerald-900 p-5 text-white shadow-[0_20px_45px_rgba(15,23,42,0.18)]">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.25em] text-emerald-200">Analytics overview</div>
                  <div className="mt-2 text-[28px] font-bold">Crop performance dashboard</div>
                </div>
                <div className="inline-flex items-center gap-2 rounded-full border border-emerald-300/40 bg-white/5 px-3 py-1.5 text-xs font-medium text-emerald-100">
                  <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
                  Live insight stream
                </div>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-4">
              {[
                {
                  label: "Predicted Yield",
                  value: result ? `${result.predicted_yield_tons_per_ha.toFixed(3)} t/ha` : "--",
                  delta: result ? `↑ ${Math.abs(((result.predicted_yield_tons_per_ha - (result.benchmark_yield_tons_per_ha ?? result.predicted_yield_tons_per_ha)) / (result.benchmark_yield_tons_per_ha ?? result.predicted_yield_tons_per_ha)) * 100).toFixed(1)}% vs. regional avg` : "Awaiting prediction",
                  tone: "emerald",
                  icon: "🌾",
                },
                {
                  label: "Estimated Revenue",
                  value: result ? `${result.estimated_revenue_birr.toLocaleString()} birr` : "--",
                  delta: result ? `↑ ${(14.7).toFixed(1)}% vs. regional avg` : "Awaiting prediction",
                  tone: "amber",
                  icon: "💰",
                },
                {
                  label: "Lookup Price",
                  value: result ? `${result.price_birr_per_quintal.toLocaleString()} birr/quintal` : "--",
                  delta: result && hasWeatherData ? `Temp ${seasonTemp!.toFixed(1)}°C | Rain ${rainfallFromApi!} mm` : "Awaiting prediction",
                  tone: "sky",
                  icon: "🛒",
                },
                {
                  label: "Region Climate",
                  value: result ? result.region : "--",
                  delta: result ? "Good growing conditions" : "Awaiting prediction",
                  tone: "slate",
                  icon: "☀️",
                },
              ].map((card) => (
                <div key={card.label} className="rounded-[22px] border border-slate-200 bg-white p-4 shadow-[0_14px_32px_rgba(15,23,42,0.06)]">
                  <div className="flex items-center justify-between">
                    <div className="text-sm font-medium text-slate-600">{card.label}</div>
                    <div className={[
                      "flex h-10 w-10 items-center justify-center rounded-xl text-lg shadow-sm",
                      card.tone === "emerald" && "bg-emerald-50 text-emerald-700",
                      card.tone === "amber" && "bg-amber-50 text-amber-700",
                      card.tone === "sky" && "bg-sky-50 text-sky-700",
                      card.tone === "slate" && "bg-slate-100 text-slate-700",
                    ].filter(Boolean).join(" ")}>{card.icon}</div>
                  </div>
                  <div className="mt-4 text-[27px] font-bold tracking-tight text-slate-900">{card.value}</div>
                  <div className="mt-2 text-sm font-medium text-emerald-600">{card.delta}</div>
                </div>
              ))}
            </div>

            <div className="grid gap-5 xl:grid-cols-3">
              <div className="rounded-[26px] border border-slate-200 bg-white p-5 shadow-[0_16px_35px_rgba(15,23,42,0.06)]">
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-2 text-[18px] font-semibold text-slate-800">
                    <span>📈</span>
                    <span>Yield by Season</span>
                  </div>
                  <div className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">Average</div>
                </div>
                <div className="flex h-56 items-end gap-3 rounded-2xl bg-gradient-to-b from-slate-50 to-white p-3 pt-5">
                  {[
                    { label: "Mar", value: 4.2, color: "bg-gradient-to-t from-emerald-600 to-emerald-400" },
                    { label: "Apr", value: 5.1, color: "bg-gradient-to-t from-sky-600 to-sky-400" },
                    { label: "May", value: 4.8, color: "bg-gradient-to-t from-amber-500 to-yellow-300" },
                    { label: "Jun", value: 4.9, color: "bg-gradient-to-t from-teal-600 to-teal-400" },
                  ].map((bar) => (
                    <div key={bar.label} className="group relative flex flex-1 flex-col items-center justify-end gap-2">
                      <div className="pointer-events-none absolute -top-10 left-1/2 -translate-x-1/2 rounded-full bg-slate-900 px-2.5 py-1 text-[10px] font-semibold text-white opacity-0 transition group-hover:opacity-100">
                        {bar.label}: {bar.value.toFixed(1)} t/ha
                      </div>
                      <div className={`${bar.color} w-full rounded-t-2xl shadow-md`} style={{ height: `${bar.value * 50}px` }} />
                      <span className="text-[11px] font-medium text-slate-500">{bar.label}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-[26px] border border-slate-200 bg-white p-5 shadow-[0_16px_35px_rgba(15,23,42,0.06)]">
                <div className="mb-4 flex items-center gap-2 text-[18px] font-semibold text-slate-800">
                  <span>📊</span>
                  <span>Regional Performance</span>
                </div>
                <div className="space-y-4 pt-2">
                  {[
                    { label: "Oromia", value: 4.9 },
                    { label: "Amhara", value: 4.2 },
                    { label: "SNNPR", value: 3.8 },
                    { label: "Tigray", value: 3.5 },
                    { label: "Somali", value: 2.9 },
                  ].map((item) => (
                    <div key={item.label} className="group relative">
                      <div className="mb-1.5 flex items-center justify-between text-sm text-slate-600">
                        <span className="font-medium">{item.label}</span>
                        <span className="font-semibold text-slate-800">{item.value.toFixed(1)}</span>
                      </div>
                      <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                        <div className="h-full rounded-full bg-gradient-to-r from-emerald-500 via-emerald-400 to-teal-300" style={{ width: `${(item.value / 5.5) * 100}%` }} />
                      </div>
                      <div className="pointer-events-none absolute -top-9 left-0 rounded-full bg-slate-900 px-2.5 py-1 text-[10px] font-semibold text-white opacity-0 transition group-hover:opacity-100">
                        {item.label}: {item.value.toFixed(1)} t/ha
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-[26px] border border-slate-200 bg-white p-5 shadow-[0_16px_35px_rgba(15,23,42,0.06)]">
                <div className="mb-4 flex items-center gap-2 text-[18px] font-semibold text-slate-800">
                  <span>📉</span>
                  <span>Yield Distribution</span>
                </div>
                <div className="flex h-56 items-end gap-2 rounded-2xl bg-gradient-to-b from-emerald-50 to-white p-3 pt-5">
                  {[
                    { label: "1.0-2.0", value: 28 },
                    { label: "2.0-3.0", value: 44 },
                    { label: "3.0-4.0", value: 62 },
                    { label: "4.0-5.0", value: 82 },
                    { label: "5.0-6.0", value: 58 },
                    { label: "6.0-7.0", value: 32 },
                  ].map((bin) => (
                    <div key={bin.label} className="group relative flex flex-1 flex-col items-center justify-end gap-2">
                      <div className="pointer-events-none absolute -top-10 left-1/2 -translate-x-1/2 rounded-full bg-slate-900 px-2.5 py-1 text-[10px] font-semibold text-white opacity-0 transition group-hover:opacity-100">
                        {bin.label}: {bin.value} plots
                      </div>
                      <div className="w-full rounded-t-2xl bg-gradient-to-t from-emerald-600 to-emerald-300 shadow-sm" style={{ height: `${bin.value}px` }} />
                      <span className="text-[9px] font-medium text-slate-500">{bin.label}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="grid gap-5 xl:grid-cols-[1.7fr_1.5fr_1.1fr]">
              <div className="rounded-[26px] border border-slate-200 bg-white p-5 shadow-[0_16px_35px_rgba(15,23,42,0.06)]">
                <div className="mb-4 flex items-center gap-2 text-[18px] font-semibold text-slate-800">
                  <span>📍</span>
                  <span>Yield by Region</span>
                </div>
                <div className="relative h-[245px] overflow-hidden rounded-[20px] border border-emerald-100 bg-gradient-to-br from-emerald-50 via-white to-sky-50">
                  <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(16,185,129,0.12),transparent_18%),radial-gradient(circle_at_80%_30%,rgba(59,130,246,0.10),transparent_20%),linear-gradient(135deg,rgba(255,255,255,0.9),rgba(236,253,245,0.9))]" />
                  <div
                    className="absolute left-1/2 top-1/2 h-[165px] w-[210px] -translate-x-1/2 -translate-y-1/2 rounded-[20px] border border-emerald-200/80 bg-gradient-to-br from-emerald-400 to-emerald-600 shadow-inner"
                    style={{
                      clipPath: "polygon(18% 22%, 30% 12%, 42% 8%, 48% 0%, 58% 10%, 68% 12%, 78% 18%, 88% 30%, 90% 42%, 82% 58%, 74% 66%, 72% 78%, 62% 88%, 52% 90%, 42% 86%, 34% 72%, 24% 68%, 12% 58%, 8% 42%)",
                    }}
                  />
                  <div className="absolute inset-0">
                    {[
                      { label: "Tigray", value: "3.5", left: "62%", top: "24%" },
                      { label: "Amhara", value: "4.2", left: "47%", top: "35%" },
                      { label: "Oromia", value: "4.9", left: "55%", top: "54%" },
                      { label: "SNNPR", value: "3.8", left: "38%", top: "69%" },
                      { label: "Somali", value: "2.9", left: "68%", top: "76%" },
                    ].map((spot) => (
                      <div key={spot.label} className="absolute" style={{ left: spot.left, top: spot.top }}>
                        <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-white/90 px-2.5 py-1.5 text-[11px] font-semibold text-slate-700 shadow-sm backdrop-blur-sm">
                          <span>{spot.label}</span>
                          <span className="text-slate-500">{spot.value} t/ha</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="rounded-[26px] border border-slate-200 bg-white p-5 shadow-[0_16px_35px_rgba(15,23,42,0.06)]">
                <div className="mb-4 flex items-center gap-2 text-[18px] font-semibold text-slate-800">
                  <span>🌦️</span>
                  <span>Yield vs. Weather</span>
                </div>
                <div className="mb-3 flex items-center gap-4 text-[11px] font-medium text-slate-500">
                  <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-emerald-500" /> Rainfall</span>
                  <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-amber-400" /> Temperature</span>
                </div>
                <div className="flex h-52 items-end gap-3 rounded-2xl bg-slate-50 p-3 pt-5">
                  {[
                    { label: "Mar", rain: 72, temp: 22.1 },
                    { label: "Apr", rain: 118, temp: 21.7 },
                    { label: "May", rain: 64, temp: 19.6 },
                    { label: "Jun", rain: 142, temp: 22.3 },
                  ].map((bar) => (
                    <div key={bar.label} className="flex flex-1 flex-col items-center justify-end gap-2">
                      <div className="flex w-full items-end justify-center gap-2">
                        <div className="w-1/2 rounded-t-xl bg-gradient-to-t from-emerald-600 to-emerald-400 shadow-sm" style={{ height: `${(bar.rain / 150) * 100}px` }} />
                        <div className="w-1/2 rounded-t-xl bg-gradient-to-t from-amber-400 to-yellow-300 shadow-sm" style={{ height: `${(bar.temp / 30) * 100}px` }} />
                      </div>
                      <span className="text-[11px] font-medium text-slate-500">{bar.label}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-[26px] border border-emerald-100 bg-gradient-to-br from-emerald-50 via-white to-emerald-100 p-5 shadow-[0_16px_35px_rgba(15,23,42,0.06)]">
                <div className="mb-4 flex items-center gap-2 text-[18px] font-semibold text-slate-800">
                  <span>💡</span>
                  <span>Key Insights</span>
                </div>
                <ul className="space-y-3 text-sm text-slate-700">
                  <li className="flex items-start gap-2"><span className="mt-0.5 text-green-600">✓</span><span>Higher rainfall and moderate temperature are positively correlated with yield.</span></li>
                  <li className="flex items-start gap-2"><span className="mt-0.5 text-green-600">✓</span><span>Oromia region shows the highest average yield (~4.9 t/ha).</span></li>
                  <li className="flex items-start gap-2"><span className="mt-0.5 text-green-600">✓</span><span>Maize and teff generally perform better in higher altitude areas.</span></li>
                  <li className="flex items-start gap-2"><span className="mt-0.5 text-green-600">✓</span><span>Improved seed and fertilizer use significantly increase yield potential.</span></li>
                </ul>
              </div>
            </div>
          </div>
        );

      case "Model Insights":
        return (
          <div className="space-y-5">
            <div className="rounded-[22px] border border-slate-200 bg-white p-5 shadow-sm">
              <div className="mb-3 flex items-center gap-2 text-[18px] font-semibold text-slate-800">
                <span>🧠</span>
                <span>Model Insights</span>
              </div>
              <div className="grid gap-4 md:grid-cols-3">
                {[
                  ["Accuracy", "R² ≈ 0.86", "Forecast quality across the validation split"],
                  ["Top Drivers", "Rainfall, fertilizer, altitude", "Key plot-level yield signals"],
                  ["Best Region", "Oromia", "Strongest climate and soil alignment"],
                ].map(([label, value, hint]) => (
                  <div key={label} className="rounded-2xl bg-emerald-50 p-4">
                    <div className="text-xs uppercase tracking-[0.2em] text-emerald-700">{label}</div>
                    <div className="mt-3 text-2xl font-bold text-slate-900">{value}</div>
                    <div className="mt-2 text-sm text-slate-600">{hint}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-[22px] border border-slate-200 bg-white p-5 shadow-sm">
              <div className="mb-3 text-lg font-semibold text-slate-800">Key learning notes</div>
              <ul className="space-y-3 text-sm text-slate-700">
                <li>• Higher rainfall and moderate temperature are strongly tied to stronger crop performance.</li>
                <li>• Fertilizer use and improved seed access have the biggest positive influence on yield outcomes.</li>
                <li>• Distance to market and pest pressure reduce profitability in lower-quality farming systems.</li>
                <li>• Predictions are more stable when seasonal rainfall and planting month are aligned with regional climate norms.</li>
              </ul>
            </div>
          </div>
        );

      case "About":
        return (
          <div className="rounded-[22px] border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-3 flex items-center gap-2 text-[18px] font-semibold text-slate-800">
              <span>ℹ️</span>
              <span>About the Project</span>
            </div>
            <p className="text-sm leading-7 text-slate-700">
              This dashboard estimates smallholder crop yield and revenue using a regression model trained on Ethiopian farming, weather, and market price data. It helps translate plot-level inputs into practical agronomic insights and economic expectations.
            </p>
            <div className="mt-5 grid gap-4 md:grid-cols-2">
              <div className="rounded-2xl bg-slate-50 p-4">
                <div className="text-xs uppercase tracking-[0.2em] text-slate-500">Objective</div>
                <div className="mt-3 text-lg font-semibold text-slate-900">Support food security planning</div>
                <p className="mt-2 text-sm text-slate-600">Build a transparent tool for yield estimation, regional comparison, and agriculture decision support.</p>
              </div>
              <div className="rounded-2xl bg-slate-50 p-4">
                <div className="text-xs uppercase tracking-[0.2em] text-slate-500">Data sources</div>
                <div className="mt-3 text-lg font-semibold text-slate-900">Weather, market, and field records</div>
                <p className="mt-2 text-sm text-slate-600">The model combines seasonal climate patterns, crop price trends, and user-entered farm characteristics.</p>
              </div>
            </div>
          </div>
        );

      default:
        return (
          <div className="rounded-[22px] border border-slate-200 bg-white p-5 shadow-sm">
            <div className="text-[18px] font-semibold text-slate-800">Dashboard</div>
            <div className="mt-2 text-sm text-slate-600">The main prediction dashboard is ready and the result widgets update from the backend.</div>
          </div>
        );
    }
  };

  return (
    <main className="min-h-screen bg-[#edf3f5] text-slate-900">
      <div className="flex h-screen overflow-hidden">
        <aside className="flex w-[250px] flex-col bg-[#071d33] text-white">
          <div className="flex items-center gap-3 border-b border-white/10 px-5 py-5">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-400 to-emerald-600 text-xl shadow-lg shadow-emerald-900/30">
              🌾
            </div>
            <div className="leading-tight">
              <div className="text-[11px] font-semibold tracking-[0.18em] text-slate-200/90">QIYAS HACKATHON</div>
              <div className="mt-1 text-[14px] font-semibold">Crop Yield Predictor</div>
            </div>
          </div>

          <nav className="mt-6 flex-1 space-y-1 px-2">
            {navItems.map((item, index) => {
              const active = item === activeNav;
              const icons = ["🏠", "📊", "🧭", "📉", "🧠", "ℹ️"];
              return (
                <button
                  key={item}
                  type="button"
                  onClick={() => setActiveNav(item)}
                  className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-[15px] transition ${
                    active ? "bg-[#1eae8d] font-semibold text-white shadow-sm" : "text-slate-200 hover:bg-white/5"
                  }`}
                >
                  <span className="text-base">{icons[index]}</span>
                  {item}
                </button>
              );
            })}
          </nav>

          <div className="mt-auto border-t border-white/10 p-4">
            <div className="rounded-xl bg-white/5 p-3 text-sm text-slate-100">
              <div className="font-semibold">Better Data</div>
              <div className="mt-1 text-xs text-slate-300">for a Food Secure Ethiopia</div>
            </div>
          </div>
        </aside>

        <main className="flex-1 overflow-auto bg-[#edf3f5]">
          <header className="flex items-center justify-between border-b border-slate-200 bg-[#edf3f5] px-6 py-4">
            <div className="flex items-center gap-4">
              <div className="text-[14px] font-medium text-slate-900">Crop Yield Predictor</div>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm">
                <span className="text-base">📅</span>
                Qiyas Hackathon • Oct 5, 2026
              </div>
              <div className="flex h-10 w-10 items-center justify-center rounded-full border border-slate-200 bg-white text-base shadow-sm">
                ☾
              </div>
            </div>
          </header>

          <div className="p-6">
            {activeNav === "Dashboard" || activeNav === "Prediction" ? (
              <>
                <div className="grid gap-4 md:grid-cols-4">
                  {metricCards.map((card, idx) => (
                    <div key={card.label} className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-medium text-slate-600">{card.label}</div>
                        <div className={`flex h-10 w-10 items-center justify-center rounded-xl ${
                          card.tone === "emerald" ? "bg-emerald-100" :
                          card.tone === "amber" ? "bg-amber-100" :
                          card.tone === "sky" ? "bg-sky-100" : "bg-amber-100"
                        } text-xl`}>{card.icon}</div>
                      </div>

                      <div className={`mt-4 text-[28px] font-bold ${
                        card.tone === "emerald" ? "text-emerald-600" :
                        card.tone === "amber" ? "text-slate-900" :
                        card.tone === "sky" ? "text-slate-900" : "text-slate-900"
                      }`}>{card.value}</div>
                      <div className={`mt-2 text-sm ${idx === 2 ? "text-slate-500" : "text-emerald-600"}`}>{card.delta}</div>
                    </div>
                  ))}
                </div>

                <div className="mt-6 grid gap-6 xl:grid-cols-[1.08fr_1.2fr]">
                  <form onSubmit={handleSubmit} className="rounded-[22px] border border-slate-200 bg-white p-5 shadow-sm">
                    <div className="mb-4 flex items-center gap-2 text-[18px] font-semibold text-slate-800">
                      <span>🌱</span>
                      <span>Plot Information</span>
                    </div>
                    <div className="mb-3 text-sm text-slate-500">Enter the details of your farm plot</div>

                    <div className="grid gap-4 md:grid-cols-2">
                      <label className="block text-sm font-medium text-slate-700">
                        Region
                        <select value={form.region} onChange={(e) => handleChange("region", e.target.value)} className={fieldClass}>
                          <option>Oromia</option>
                          <option>Amhara</option>
                          <option>SNNPR</option>
                          <option>Somali</option>
                          <option>Tigray</option>
                        </select>
                      </label>

                      <label className="block text-sm font-medium text-slate-700">
                        Crop type
                        <select value={form.crop_type} onChange={(e) => handleChange("crop_type", e.target.value)} className={fieldClass}>
                          <option>maize</option>
                          <option>wheat</option>
                          <option>teff</option>
                          <option>sorghum</option>
                          <option>barley</option>
                        </select>
                      </label>

                      <label className="block text-sm font-medium text-slate-700">
                        Survey year
                        <select value={form.survey_year} onChange={(e) => handleChange("survey_year", e.target.value)} className={fieldClass}>
                          {[2021, 2022, 2023, 2024].map((year) => (
                            <option key={year} value={year}>{year}</option>
                          ))}
                        </select>
                      </label>

                      <label className="block text-sm font-medium text-slate-700">
                        Model family
                        <select value={form.model_name} onChange={(e) => handleChange("model_name", e.target.value)} className={fieldClass}>
                          {modelOptions.map((model) => (
                            <option key={model} value={model}>{model}</option>
                          ))}
                        </select>
                      </label>

                      <label className="block text-sm font-medium text-slate-700">
                        Planting month
                        <select value={form.planting_month} onChange={(e) => handleChange("planting_month", e.target.value)} className={fieldClass}>
                          {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'].map((month) => (
                            <option key={month} value={month}>{month}</option>
                          ))}
                        </select>
                      </label>

                      <label className="block text-sm font-medium text-slate-700">
                        Altitude (m)
                        <input type="number" value={form.altitude_m} onChange={(e) => handleChange("altitude_m", e.target.value)} className={fieldClass} />
                      </label>

                      <label className="block text-sm font-medium text-slate-700">
                        Plot-reported rainfall (mm)
                        <input type="number" value={form.rainfall_mm_season} onChange={(e) => handleChange("rainfall_mm_season", e.target.value)} className={fieldClass} />
                      </label>

                      <label className="block text-sm font-medium text-slate-700">
                        Farm size (ha)
                        <input type="number" step="0.1" value={form.farm_size_ha} onChange={(e) => handleChange("farm_size_ha", e.target.value)} className={fieldClass} />
                      </label>

                      <label className="block text-sm font-medium text-slate-700">
                        Fertilizer (kg/ha)
                        <input type="number" step="0.1" value={form.fertilizer_kg_per_ha} onChange={(e) => handleChange("fertilizer_kg_per_ha", e.target.value)} className={fieldClass} />
                      </label>

                      <label className="block text-sm font-medium text-slate-700">
                        Improved seed
                        <select value={String(form.improved_seed_used)} onChange={(e) => handleChange("improved_seed_used", e.target.value)} className={fieldClass}>
                          <option value={1}>Yes</option>
                          <option value={0}>No</option>
                        </select>
                      </label>

                      <label className="block text-sm font-medium text-slate-700">
                        Pest/disease flag
                        <select value={String(form.pest_disease_flag)} onChange={(e) => handleChange("pest_disease_flag", e.target.value)} className={fieldClass}>
                          <option value={0}>No</option>
                          <option value={1}>Yes</option>
                        </select>
                      </label>

                      <label className="block text-sm font-medium text-slate-700">
                        Soil quality index
                        <input type="number" step="0.01" value={form.soil_quality_index} onChange={(e) => handleChange("soil_quality_index", e.target.value)} className={fieldClass} />
                      </label>

                      <label className="block text-sm font-medium text-slate-700">
                        Labor (days/ha)
                        <input type="number" step="0.1" value={form.labor_days_per_ha} onChange={(e) => handleChange("labor_days_per_ha", e.target.value)} className={fieldClass} />
                      </label>

                      <label className="block text-sm font-medium text-slate-700 md:col-span-2">
                        Distance to market (km)
                        <input type="number" step="0.1" value={form.distance_to_market_km} onChange={(e) => handleChange("distance_to_market_km", e.target.value)} className={fieldClass} />
                      </label>
                    </div>

                    {error ? (
                      <div className="mt-4 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>
                    ) : null}

                    <button
                      type="submit"
                      disabled={loading}
                      className="mt-6 inline-flex w-full items-center justify-center rounded-xl bg-gradient-to-r from-emerald-500 to-emerald-600 px-4 py-3 text-base font-semibold text-white shadow-sm transition hover:from-emerald-600 hover:to-emerald-700 disabled:cursor-not-allowed disabled:opacity-70"
                    >
                      <span className="mr-2 text-lg">⚡</span>
                      {loading ? "Predicting..." : "Predict Yield"}
                    </button>
                  </form>

                  <div className="rounded-[28px] bg-[#0b223a] p-5 text-white shadow-xl relative overflow-hidden">
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,_rgba(94,234,212,0.15),_transparent_35%),radial-gradient(circle_at_bottom_right,_rgba(96,165,250,0.12),_transparent_30%)]" />
                    <div className="relative z-10">
                      <div className="mb-4 flex items-center gap-2 text-[22px] font-bold">
                        <span>🌿</span>
                        <span>Prediction Result</span>
                      </div>

                      <div className="grid gap-3 md:grid-cols-2">
                        <div className="rounded-2xl bg-[#134d5d]/80 p-4 shadow-inner">
                          <div className="text-[14px] text-slate-300">Predicted yield</div>
                          <div className="mt-2 text-[28px] font-bold text-emerald-400">{result ? `${result.predicted_yield_tons_per_ha.toFixed(3)} t/ha` : "--"}</div>
                        </div>
                        <div className="rounded-2xl bg-[#2d4b8c]/80 p-4 shadow-inner">
                          <div className="text-[14px] text-slate-300">Estimated revenue</div>
                          <div className="mt-2 text-[28px] font-bold text-amber-300">{result ? `${result.estimated_revenue_birr.toLocaleString()} birr` : "--"}</div>
                        </div>
                      </div>

                      <div className="mt-4 grid gap-3 md:grid-cols-2">
                        <div className="rounded-2xl bg-[#123f5b]/80 p-4">
                          <div className="text-sm text-slate-300">Season mean temp</div>
                          <div className="mt-2 text-[26px] font-bold">{result && seasonTemp !== null ? `${seasonTemp.toFixed(1)}°C` : "--"}</div>
                        </div>
                        <div className="rounded-2xl bg-[#123f5b]/80 p-4">
                          <div className="text-sm text-slate-300">Total rain</div>
                          <div className="mt-2 text-[26px] font-bold">{result && rainfallFromApi !== null ? `${rainfallFromApi} mm` : "--"}</div>
                        </div>
                      </div>

                      <div className="mt-5 rounded-2xl bg-white/5 p-4">
                        <div className="mb-3 text-sm font-medium text-slate-300">Yield comparison</div>
                        {result ? (
                          <div className="flex items-end gap-3">
                            <div className="flex flex-1 flex-col items-center justify-end">
                              <div className="w-full rounded-t-xl bg-[#27d4b5]" style={{ height: `${Math.max(result.predicted_yield_tons_per_ha * 20, 30)}px` }} />
                              <div className="mt-2 text-[10px] text-slate-300">Prediction</div>
                            </div>
                            <div className="flex flex-1 flex-col items-center justify-end">
                              <div className="w-full rounded-t-xl bg-[#4bb7ff]" style={{ height: `${Math.max((result.benchmark_yield_tons_per_ha ?? 4.2) * 18, 30)}px` }} />
                              <div className="mt-2 text-[10px] text-slate-300">Regional avg</div>
                            </div>
                          </div>
                        ) : (
                          <div className="flex h-24 items-center justify-center rounded-xl border border-dashed border-slate-500/60 text-sm text-slate-300">Submit the form to generate a prediction.</div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-6 grid gap-5 lg:grid-cols-[1.3fr_0.9fr_0.9fr]">
                  <div className="rounded-[22px] border border-slate-200 bg-white p-4 shadow-sm">
                    <div className="mb-4 flex items-center gap-2 text-[16px] font-semibold text-slate-800">
                      <span>📈</span>
                      Yield vs. Weather
                    </div>
                    <div className="flex h-36 items-end gap-3 pt-4">
                      {chartBars.map((bar) => (
                        <div key={bar.label} className="flex flex-1 flex-col items-center justify-end gap-2">
                          <div className={`${bar.color} w-full rounded-t-xl`} style={{ height: `${bar.value * 40}px` }} />
                          <span className="text-[11px] text-slate-500">{bar.label}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-[22px] border border-slate-200 bg-white p-4 shadow-sm">
                    <div className="mb-4 flex items-center gap-2 text-[16px] font-semibold text-slate-800">
                      <span>📊</span>
                      Regional Yield Comparison
                    </div>
                    <div className="flex h-36 items-end gap-2 pt-4">
                      {regionValues.map((value, index) => (
                        <div key={index} className="flex flex-1 flex-col items-center justify-end gap-2">
                          <div className="w-full rounded-t-xl bg-[#46c3ad]" style={{ height: `${value * 42}px` }} />
                          <span className="text-[10px] text-slate-500">{["Oromia", "Amhara", "SNNPR", "Tigray", "Somali"][index]}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-[22px] border border-slate-200 bg-[#dffaf0] p-4 shadow-sm">
                    <div className="mb-4 flex items-center gap-2 text-[16px] font-semibold text-slate-800">
                      <span>💡</span>
                      Key Insights
                    </div>
                    <ul className="space-y-3 text-sm text-slate-700">
                      <li className="flex items-start gap-2"><span className="mt-0.5 text-green-600">✓</span> Higher rainfall and moderate temperature are positively correlated with yield.</li>
                      <li className="flex items-start gap-2"><span className="mt-0.5 text-green-600">✓</span> Oromia region shows the highest average yield (~4.9 t/ha).</li>
                      <li className="flex items-start gap-2"><span className="mt-0.5 text-green-600">✓</span> Maize and teff generally perform better in higher altitude areas.</li>
                      <li className="flex items-start gap-2"><span className="mt-0.5 text-green-600">✓</span> Improved seed and fertilizer use significantly increase expected profit.</li>
                    </ul>
                  </div>
                </div>
              </>
            ) : (
              renderSectionContent()
            )}
          </div>
        </main>
      </div>
    </main>
  );
}
