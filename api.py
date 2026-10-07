"""FastAPI phục vụ mô hình churn.  Chạy local: uvicorn api:app --reload

  /          Giao diện web để nhập thông tin khách hàng và xem dự đoán
  /predict   API dự đoán 1 khách        /predict_batch  API dự đoán nhiều khách
  /explain   Giải thích 1 dự đoán (SHAP) /health         Kiểm tra trạng thái
  /docs      Swagger UI (dành cho lập trình viên)
"""
import warnings
from contextlib import asynccontextmanager
from typing import List, Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", message="X does not have valid feature names")

MODEL_PATH = "model/churn_pipeline.joblib"
FEATURES = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)   # nạp 1 lần khi khởi động
    yield
    state.clear()


app = FastAPI(title="Bank Customer Churn API", version="1.1", lifespan=lifespan)


class Customer(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850, examples=[650])
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Female", "Male"]
    Age: int = Field(..., ge=18, le=100, examples=[40])
    Tenure: int = Field(..., ge=0, le=10, examples=[5])
    Balance: float = Field(..., ge=0, examples=[0.0])
    NumOfProducts: int = Field(..., ge=1, le=4, examples=[2])
    HasCrCard: Literal[0, 1]
    IsActiveMember: Literal[0, 1]
    EstimatedSalary: float = Field(..., ge=0, examples=[100000.0])


class Prediction(BaseModel):
    churn_probability: float
    high_risk: bool
    threshold: float


class Contribution(BaseModel):
    feature: str
    contribution: float          # log-odds; dương = đẩy về phía rời đi


class Explanation(BaseModel):
    churn_probability: float
    baseline_log_odds: float
    contributions: List[Contribution]   # đã sắp xếp theo mức tác động giảm dần


def _predict(customers: List[Customer], threshold: float) -> List[Prediction]:
    df = pd.DataFrame([c.model_dump() for c in customers])[FEATURES]
    proba = state["model"].predict_proba(df)[:, 1]
    return [Prediction(churn_probability=round(float(p), 4),
                       high_risk=bool(p >= threshold), threshold=threshold) for p in proba]


def _explain(customer: Customer) -> Explanation:
    """SHAP của LightGBM (pred_contrib). baseline + tổng đóng góp = log-odds của dự đoán."""
    pipe = state["model"]
    df = pd.DataFrame([customer.model_dump()])[FEATURES]
    contrib = pipe[-1].predict(pipe[:-1].transform(df), pred_contrib=True)[0]
    names = [n.split("__", 1)[1] for n in pipe[:-1].get_feature_names_out()]
    raw = dict(zip(names, (float(v) for v in contrib[:-1])))
    # gộp các cột one-hot về đúng tên trường nhập vào
    merged = {k: v for k, v in raw.items() if not k.startswith(("Geography_", "Gender_"))}
    merged["Geography"] = raw["Geography_Germany"] + raw["Geography_Spain"]
    merged["Gender"] = raw["Gender_Male"]
    items = sorted(merged.items(), key=lambda kv: abs(kv[1]), reverse=True)
    proba = float(pipe.predict_proba(df)[:, 1][0])
    return Explanation(
        churn_probability=round(proba, 4),
        baseline_log_odds=round(float(contrib[-1]), 4),
        contributions=[Contribution(feature=k, contribution=round(v, 4)) for k, v in items],
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, threshold: float = 0.5):
    return _predict([customer], threshold)[0]


@app.post("/predict_batch", response_model=List[Prediction])
def predict_batch(customers: List[Customer], threshold: float = 0.5):
    return _predict(customers, threshold)


@app.post("/explain", response_model=Explanation)
def explain(customer: Customer):
    return _explain(customer)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return HTML


HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bank Customer Churn Prediction</title>
<style>
  :root{
    --bg:#f5f6fa; --card:#ffffff; --text:#1c2230; --muted:#667085; --line:#e4e7ee;
    --accent:#2563eb; --accent-d:#1d4fc4; --bad:#dc2626; --bad-bg:#fef2f2;
    --good:#16a34a; --good-bg:#f0fdf4; --track:#e9ecf3;
  }
  @media (prefers-color-scheme: dark){
    :root{
      --bg:#0f1218; --card:#181c25; --text:#e8ebf2; --muted:#98a2b3; --line:#2a303c;
      --accent:#5b8def; --accent-d:#7aa2f7; --bad:#f87171; --bad-bg:#2a1618;
      --good:#4ade80; --good-bg:#12261a; --track:#2a303c;
    }
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
       font-family:system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;line-height:1.5}
  .wrap{max-width:1040px;margin:0 auto;padding:24px 16px 48px}
  header{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;margin-bottom:20px}
  h1{font-size:1.5rem;margin:0 0 4px}
  .sub{color:var(--muted);margin:0;font-size:.95rem}
  .lang button{border:1px solid var(--line);background:var(--card);color:var(--text);
       padding:6px 12px;cursor:pointer;font-size:.85rem}
  .lang button:first-child{border-radius:8px 0 0 8px}
  .lang button:last-child{border-radius:0 8px 8px 0;border-left:0}
  .lang button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
  @media (max-width:820px){.grid{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}
  .card h2{font-size:1.05rem;margin:0 0 14px}
  .fields{display:grid;grid-template-columns:1fr 1fr;gap:12px 14px}
  .full{grid-column:1/-1}
  label{display:block;font-size:.85rem;font-weight:600;margin-bottom:4px}
  label small{font-weight:400;color:var(--muted)}
  input,select{width:100%;padding:9px 10px;border:1px solid var(--line);border-radius:8px;
       background:var(--bg);color:var(--text);font-size:.95rem}
  input:focus,select:focus{outline:2px solid var(--accent);outline-offset:0}
  .row{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap}
  button.primary{background:var(--accent);color:#fff;border:0;border-radius:8px;
       padding:11px 20px;font-size:1rem;font-weight:600;cursor:pointer;flex:1}
  button.primary:hover{background:var(--accent-d)}
  button.primary:disabled{opacity:.6;cursor:wait}
  button.ghost{background:transparent;color:var(--text);border:1px solid var(--line);
       border-radius:8px;padding:9px 12px;font-size:.85rem;cursor:pointer}
  button.ghost:hover{border-color:var(--accent)}
  .presets{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
  .thr{display:flex;align-items:center;gap:10px}
  .thr input{padding:0}
  .empty{color:var(--muted);text-align:center;padding:48px 8px}
  .verdict{border-radius:12px;padding:16px;text-align:center;margin-bottom:16px}
  .verdict.high{background:var(--bad-bg);color:var(--bad)}
  .verdict.low{background:var(--good-bg);color:var(--good)}
  .pct{font-size:2.6rem;font-weight:800;line-height:1.1}
  .lbl{font-weight:600;margin-top:2px}
  .meter{position:relative;height:12px;border-radius:6px;background:var(--track);margin:6px 0 4px}
  .meter .fill{position:absolute;left:0;top:0;bottom:0;border-radius:6px}
  .meter .mark{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--text)}
  .scale{display:flex;justify-content:space-between;font-size:.75rem;color:var(--muted)}
  .why{margin-top:18px}
  .why h3{font-size:.95rem;margin:0 0 4px}
  .why p{font-size:.8rem;color:var(--muted);margin:0 0 10px}
  .f{display:grid;grid-template-columns:150px 1fr 52px;gap:8px;align-items:center;margin:6px 0;font-size:.85rem}
  .f .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .bar{position:relative;height:14px}
  .bar::before{content:"";position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--line)}
  .bar i{position:absolute;top:2px;bottom:2px;border-radius:3px}
  .bar i.up{left:50%;background:var(--bad)}
  .bar i.dn{right:50%;background:var(--good)}
  .f .v{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
  .legend{display:flex;gap:14px;font-size:.75rem;color:var(--muted);margin-top:8px}
  .legend b{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
  .err{background:var(--bad-bg);color:var(--bad);border-radius:8px;padding:12px;font-size:.9rem}
  .note{font-size:.8rem;color:var(--muted);margin-top:16px}
  footer{margin-top:24px;font-size:.8rem;color:var(--muted);text-align:center}
  footer a{color:var(--accent)}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1 data-i18n="title"></h1>
      <p class="sub" data-i18n="subtitle"></p>
    </div>
    <div class="lang">
      <button id="bt-vi" type="button" class="on">Tiếng Việt</button><button id="bt-en" type="button">English</button>
    </div>
  </header>

  <div class="grid">
    <section class="card">
      <h2 data-i18n="formTitle"></h2>
      <div class="presets">
        <button class="ghost" type="button" id="p-low" data-i18n="presetLow"></button>
        <button class="ghost" type="button" id="p-mid" data-i18n="presetMid"></button>
        <button class="ghost" type="button" id="p-high" data-i18n="presetHigh"></button>
      </div>
      <form id="f" autocomplete="off">
        <div class="fields">
          <div>
            <label for="Age"><span data-i18n="Age"></span> <small>(18–100)</small></label>
            <input id="Age" type="number" min="18" max="100" step="1" required value="40">
          </div>
          <div>
            <label for="NumOfProducts" data-i18n="NumOfProducts"></label>
            <select id="NumOfProducts">
              <option value="1">1</option><option value="2" selected>2</option>
              <option value="3">3</option><option value="4">4</option>
            </select>
          </div>
          <div>
            <label for="Geography" data-i18n="Geography"></label>
            <select id="Geography">
              <option>France</option><option>Germany</option><option>Spain</option>
            </select>
          </div>
          <div>
            <label for="Gender" data-i18n="Gender"></label>
            <select id="Gender">
              <option value="Female" data-i18n="Female"></option>
              <option value="Male" data-i18n="Male"></option>
            </select>
          </div>
          <div>
            <label for="IsActiveMember" data-i18n="IsActiveMember"></label>
            <select id="IsActiveMember">
              <option value="1" data-i18n="yes"></option><option value="0" data-i18n="no"></option>
            </select>
          </div>
          <div>
            <label for="HasCrCard" data-i18n="HasCrCard"></label>
            <select id="HasCrCard">
              <option value="1" data-i18n="yes"></option><option value="0" data-i18n="no"></option>
            </select>
          </div>
          <div>
            <label for="CreditScore"><span data-i18n="CreditScore"></span> <small>(300–850)</small></label>
            <input id="CreditScore" type="number" min="300" max="850" step="1" required value="650">
          </div>
          <div>
            <label for="Tenure"><span data-i18n="Tenure"></span> <small data-i18n="years0_10"></small></label>
            <input id="Tenure" type="number" min="0" max="10" step="1" required value="5">
          </div>
          <div>
            <label for="Balance" data-i18n="Balance"></label>
            <input id="Balance" type="number" min="0" step="any" required value="0">
          </div>
          <div>
            <label for="EstimatedSalary" data-i18n="EstimatedSalary"></label>
            <input id="EstimatedSalary" type="number" min="0" step="any" required value="100000">
          </div>
          <div class="full">
            <label for="thr"><span data-i18n="threshold"></span>: <b id="thrv">0.50</b></label>
            <div class="thr"><input id="thr" type="range" min="0.1" max="0.9" step="0.05" value="0.5"></div>
            <small style="color:var(--muted)" data-i18n="thresholdHint"></small>
          </div>
        </div>
        <div class="row">
          <button class="primary" id="go" type="submit" data-i18n="predict"></button>
        </div>
      </form>
    </section>

    <section class="card" aria-live="polite">
      <h2 data-i18n="resultTitle"></h2>
      <div id="out"><div class="empty" data-i18n="empty"></div></div>
    </section>
  </div>

  <p class="note" data-i18n="disclaimer"></p>
  <footer>
    <a href="/docs">API docs (Swagger)</a> · <span data-i18n="footer"></span>
  </footer>
</div>

<script>
const I18N = {
  vi:{
    title:"Dự đoán khách hàng ngân hàng rời bỏ",
    subtitle:"Nhập thông tin khách hàng để ước tính xác suất họ sẽ rời ngân hàng (mô hình LightGBM).",
    formTitle:"Thông tin khách hàng", resultTitle:"Kết quả dự đoán",
    presetLow:"Ví dụ: rủi ro thấp", presetMid:"Ví dụ: trung bình", presetHigh:"Ví dụ: rủi ro cao",
    Age:"Tuổi", NumOfProducts:"Số sản phẩm đang dùng", Geography:"Quốc gia", Gender:"Giới tính",
    IsActiveMember:"Khách có hoạt động thường xuyên?", HasCrCard:"Có thẻ tín dụng?",
    CreditScore:"Điểm tín dụng", Tenure:"Số năm gắn bó", years0_10:"(0–10 năm)",
    Balance:"Số dư tài khoản", EstimatedSalary:"Lương ước tính",
    Female:"Nữ", Male:"Nam", yes:"Có", no:"Không",
    threshold:"Ngưỡng cảnh báo", thresholdHint:"Xác suất từ ngưỡng này trở lên sẽ bị đánh dấu là rủi ro cao.",
    predict:"Dự đoán", loading:"Đang tính… (lần đầu có thể mất tới 1 phút vì máy chủ miễn phí cần khởi động)",
    empty:"Điền thông tin ở bên trái rồi bấm “Dự đoán”.",
    high:"Rủi ro CAO", low:"Chưa vượt ngưỡng cảnh báo",
    probLbl:"xác suất rời đi", thresholdMark:"ngưỡng",
    whyTitle:"Vì sao có kết quả này?",
    whyDesc:"Mức đóng góp của từng yếu tố (SHAP). Thanh đỏ làm tăng nguy cơ rời đi, thanh xanh làm giảm.",
    up:"tăng nguy cơ", dn:"giảm nguy cơ",
    disclaimer:"Lưu ý: mô hình huấn luyện trên dữ liệu tổng hợp (Kaggle) nên chỉ mang tính minh họa, không dùng để ra quyết định thực tế. Phần giải thích cho biết mô hình dùng yếu tố nào, không chứng minh quan hệ nhân quả.",
    footer:"Nam Huy · Customer Churn Prediction",
    err:"Không gọi được API", errInput:"Dữ liệu chưa hợp lệ. Hãy kiểm tra lại các ô."
  },
  en:{
    title:"Bank Customer Churn Prediction",
    subtitle:"Enter a customer's details to estimate the probability they will leave the bank (LightGBM model).",
    formTitle:"Customer details", resultTitle:"Prediction",
    presetLow:"Example: low risk", presetMid:"Example: medium", presetHigh:"Example: high risk",
    Age:"Age", NumOfProducts:"Number of products", Geography:"Country", Gender:"Gender",
    IsActiveMember:"Active member?", HasCrCard:"Has a credit card?",
    CreditScore:"Credit score", Tenure:"Tenure", years0_10:"(0–10 years)",
    Balance:"Account balance", EstimatedSalary:"Estimated salary",
    Female:"Female", Male:"Male", yes:"Yes", no:"No",
    threshold:"Alert threshold", thresholdHint:"A probability at or above this value is flagged as high risk.",
    predict:"Predict", loading:"Calculating… (the first request can take up to a minute while the free server wakes up)",
    empty:"Fill in the form on the left and press “Predict”.",
    high:"HIGH risk", low:"Below the alert threshold",
    probLbl:"churn probability", thresholdMark:"threshold",
    whyTitle:"Why this result?",
    whyDesc:"Contribution of each factor (SHAP). Red bars raise the risk of leaving, green bars lower it.",
    up:"raises risk", dn:"lowers risk",
    disclaimer:"Note: the model is trained on synthetic data (Kaggle), so this is a demo and not for real decisions. The explanation shows what the model uses, not cause and effect.",
    footer:"Nam Huy · Customer Churn Prediction",
    err:"Could not reach the API", errInput:"Invalid input. Please check the fields."
  }
};
const PRESETS = {
  low:{CreditScore:720,Geography:"France",Gender:"Male",Age:32,Tenure:6,Balance:0,NumOfProducts:2,HasCrCard:1,IsActiveMember:1,EstimatedSalary:85000},
  mid:{CreditScore:380,Geography:"Germany",Gender:"Male",Age:43,Tenure:3,Balance:150000,NumOfProducts:2,HasCrCard:0,IsActiveMember:0,EstimatedSalary:100000},
  high:{CreditScore:600,Geography:"Germany",Gender:"Female",Age:52,Tenure:4,Balance:120000,NumOfProducts:1,HasCrCard:1,IsActiveMember:0,EstimatedSalary:90000}
};
const IDS = ["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"];
const $ = id => document.getElementById(id);
let lang = "vi", last = null;
const t = k => I18N[lang][k];

function applyLang(){
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach(e => { e.textContent = t(e.dataset.i18n); });
  $("bt-vi").classList.toggle("on", lang === "vi");
  $("bt-en").classList.toggle("on", lang === "en");
  if (last) render(last.pred, last.exp, last.payload);
}
function fill(p){ IDS.forEach(k => { $(k).value = p[k]; }); }
function payload(){
  const p = {};
  IDS.forEach(k => { const v = $(k).value; p[k] = (k==="Geography"||k==="Gender") ? v : Number(v); });
  return p;
}
function valueLabel(k, p){
  const v = p[k];
  if (k==="Gender") return t(v);
  if (k==="HasCrCard"||k==="IsActiveMember") return v ? t("yes") : t("no");
  if (k==="Balance"||k==="EstimatedSalary") return Number(v).toLocaleString("en-US");
  return String(v);
}
function render(pred, exp, p){
  const pct = (pred.churn_probability*100);
  const high = pred.high_risk;
  const thr = pred.threshold*100;
  const color = high ? "var(--bad)" : "var(--good)";
  const rows = exp.contributions.filter(c => Math.abs(c.contribution) >= 0.05).slice(0,7);
  const max = Math.max(...rows.map(c => Math.abs(c.contribution)), 0.01);
  const list = rows.map(c => {
    const w = (Math.abs(c.contribution)/max*50).toFixed(1);
    const cls = c.contribution >= 0 ? "up" : "dn";
    const sign = c.contribution >= 0 ? "+" : "−";
    return `<div class="f"><div class="nm" title="${t(c.feature)}">${t(c.feature)} = ${valueLabel(c.feature,p)}</div>
      <div class="bar"><i class="${cls}" style="width:${w}%"></i></div>
      <div class="v">${sign}${Math.abs(c.contribution).toFixed(2)}</div></div>`;
  }).join("");
  $("out").innerHTML = `
    <div class="verdict ${high?"high":"low"}">
      <div class="pct">${pct.toFixed(1)}%</div>
      <div>${t("probLbl")}</div>
      <div class="lbl">${high ? "⚠️ "+t("high") : "✅ "+t("low")}</div>
    </div>
    <div class="meter"><div class="fill" style="width:${pct}%;background:${color}"></div>
      <div class="mark" style="left:${thr}%" title="${t("thresholdMark")} ${thr.toFixed(0)}%"></div></div>
    <div class="scale"><span>0%</span><span>${t("thresholdMark")}: ${thr.toFixed(0)}%</span><span>100%</span></div>
    <div class="why"><h3>${t("whyTitle")}</h3><p>${t("whyDesc")}</p>${list}
      <div class="legend"><span><b style="background:var(--bad)"></b>${t("up")}</span><span><b style="background:var(--good)"></b>${t("dn")}</span></div>
    </div>`;
}
async function post(url, body){
  const r = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  if (!r.ok){ const e = new Error("http"); e.status = r.status; throw e; }
  return r.json();
}
$("f").addEventListener("submit", async ev => {
  ev.preventDefault();
  const p = payload(), thr = $("thr").value, btn = $("go");
  btn.disabled = true;
  $("out").innerHTML = `<div class="empty">${t("loading")}</div>`;
  try{
    const [pred, exp] = await Promise.all([post("/predict?threshold="+thr, p), post("/explain", p)]);
    last = {pred, exp, payload:p};
    render(pred, exp, p);
  }catch(e){
    $("out").innerHTML = `<div class="err">${e.status===422 ? t("errInput") : t("err")}</div>`;
  }finally{ btn.disabled = false; }
});
$("thr").addEventListener("input", e => { $("thrv").textContent = Number(e.target.value).toFixed(2); });
$("p-low").onclick  = () => fill(PRESETS.low);
$("p-mid").onclick  = () => fill(PRESETS.mid);
$("p-high").onclick = () => fill(PRESETS.high);
$("bt-vi").onclick = () => { lang = "vi"; applyLang(); };
$("bt-en").onclick = () => { lang = "en"; applyLang(); };
applyLang();
</script>
</body>
</html>
"""
