const express = require('express');
const WebSocket = require('ws');
const cors = require('cors');
const OpenAI = require('openai');
const Anthropic = require('@anthropic-ai/sdk');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

let ragData = {
  books: [], afpk: [], bantoe: [], quotes: [], keywords: {},
  questions: [], workbook: [], consultation: [], lecture: [],
  cfha: [], custQ: [], nagging: [],
};

function loadRAGData() {
  const dataDir = __dirname;
  const fileMap = {
    books:        'rag_chunks.json',
    afpk:         'afpk_knowledge_base.json',
    bantoe:       'bantoe_cases_436.json',
    consultation: 'consultation_chunks.json',
    lecture:      'lecture_chunks.json',
    workbook:     'workbook_chunks.json',
    cfha:         'cfha_script_chunks.json',
    quotes:       'quotes_100.json',
    nagging:      'nagging_100.json',
    custQ:        'customer_questions_100.json',
  };
  let total = 0;
  for (const [key, fileName] of Object.entries(fileMap)) {
    try {
      const filePath = path.join(dataDir, fileName);
      if (!fs.existsSync(filePath)) { console.log('[RAG] ⚠️ ' + fileName + ' 없음 — 건너뜀'); continue; }
      const raw = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
      ragData[key] = Array.isArray(raw) ? raw : (raw.chunks || raw.items || raw.data || []);
      console.log('[RAG] ✅ ' + fileName + ': ' + ragData[key].length + '개');
      total += ragData[key].length;
    } catch (e) { console.error('[RAG] ❌ ' + fileName + ' 로드 실패:', e.message); }
  }
  console.log('[RAG] ━━━ 총 ' + total + '개 청크 로드 완료 ━━━');
}

function searchRAG(query, topK = 3) {
  if (!query) return [];
  const q = query.toLowerCase();
  const words = q.split(/\s+/).filter(w => w.length >= 2);
  const results = [];
  function score(chunk, titleField, contentField, label, titleBonus = 3) {
    const title   = (chunk[titleField]   || '').toLowerCase();
    const content = (chunk[contentField] || '').toLowerCase();
    const kws     = (chunk.keywords || []).join(' ').toLowerCase();
    let s = 0;
    for (const w of words) {
      if (title.includes(w))   s += titleBonus;
      if (content.includes(w)) s += 2;
      if (kws.includes(w))     s += 1;
    }
    if (s > 0) results.push({ source: label, score: s, topic: chunk[titleField] || '', content: (chunk[contentField] || '').slice(0, 500) });
  }
  for (const q2 of ragData.custQ) {
    const text = ((q2.question||'') + ' ' + (q2.answer||'')).toLowerCase();
    let s = 0;
    for (const w of words) if (text.includes(w)) s += 2;
    if (s > 0) results.push({ source: '고객Q&A', score: s, topic: q2.question||'', content: `Q: ${q2.question||''}\nA: ${q2.answer||''}`.slice(0,500) });
  }
  for (const n of ragData.nagging) {
    const text = (n.nagging||n.content||'').toLowerCase();
    let s = 0;
    for (const w of words) if (text.includes(w)) s += 2;
    if (s > 0) results.push({ source: '금융잔소리', score: s, topic: n.category||'', content: (n.nagging||n.content||'').slice(0,300) });
  }
  for (const c of ragData.books)        score(c, 'title',  'content', '저서');
  for (const c of ragData.afpk)         score(c, 'topic',  'content', 'AFPK');
  for (const c of ragData.bantoe)       score(c, 'title',  'content', '반퇴시대', 2);
  for (const c of ragData.workbook)     score(c, 'topic',  'content', '워크북');
  for (const c of ragData.consultation) score(c, 'source', 'content', '상담사례', 2);
  for (const c of ragData.lecture)      score(c, 'source', 'content', '전문강의');
  for (const c of ragData.cfha)         score(c, 'source', 'content', 'CFHA');
  if (ragData.quotes.length > 0) {
    const rq = ragData.quotes[Math.floor(Math.random() * ragData.quotes.length)];
    results.push({ source: '명언', score: 0.5, topic: '금융명언', content: rq.quote || rq.content || '' });
  }
  return results.sort((a, b) => b.score - a.score).slice(0, topK);
}

let formulaChunks = [];

function loadFormulaRAG() {
  try {
    const filePath = path.join(__dirname, 'rag_formulas.json');
    if (!fs.existsSync(filePath)) { console.log('[RAG-공식] ⚠️  rag_formulas.json 없음 — 건너뜀'); return; }
    const data    = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    formulaChunks = data.chunks || [];
    console.log(`[RAG-공식] ✅ ${formulaChunks.length}개 공식 청크 로드 완료`);
  } catch (e) { console.error('[RAG-공식] ❌ 로드 실패:', e.message); formulaChunks = []; }
}

function searchFormulaRAG(query, maxResults = 2) {
  if (!formulaChunks.length || !query) return [];
  const tokens = query.replace(/[^\w가-힣]/g, ' ').split(/\s+/).filter(t => t.length >= 2);
  if (!tokens.length) return [];
  const scored = formulaChunks.map(chunk => {
    const text = (chunk.content + ' ' + (chunk.keywords || []).join(' ')).toLowerCase();
    let s = 0;
    tokens.forEach(t => {
      s += ((text.match(new RegExp(t.toLowerCase(), 'g')) || []).length) * 2;
      if (chunk.name.toLowerCase().includes(t.toLowerCase())) s += 4;
    });
    return { ...chunk, _s: s };
  });
  return scored.filter(c => c._s > 0).sort((a, b) => b._s - a._s).slice(0, maxResults).map(({ _s, ...c }) => c);
}

function buildFormulaContext(results) {
  if (!results || !results.length) return '';
  let ctx = '\n[오상열 CFP 재무설계 공식]\n';
  results.forEach((c, i) => {
    ctx += `${i + 1}. ${c.name}: ${c.raw.formula}\n`;
    const detail = c.raw.details.length > 180 ? c.raw.details.slice(0, 180) + '…' : c.raw.details;
    ctx += `   (${detail})\n`;
  });
  return ctx;
}

loadRAGData();
loadFormulaRAG();

const createSpendingPrompt = (userName, financialContext, budgetInfo, ragContext = '') => {
  const name = financialContext?.name || userName || '고객';
  const age = financialContext?.age || 0;
  const monthlyIncome = financialContext?.monthlyIncome || 0;
  const dailyBudget = budgetInfo?.dailyBudget || financialContext?.dailyBudget || 0;
  const todaySpent = budgetInfo?.todaySpent || financialContext?.todaySpent || 0;
  const remainingBudget = budgetInfo?.remainingBudget || financialContext?.remainingBudget || 0;
  const ragSection = ragContext ? `\n## 참고 지식 (RAG)\n${ragContext}\n` : '';

  return `당신은 "머니야"입니다. ${name}님의 AI 지출관리 코치입니다.

## 호출 규칙 (최우선!)
- "${name}" 또는 "머니야"라고 부르면: "네, ${name}님!" 이것만 말하고 멈추세요
- 절대 추가 설명하지 마세요

## 말투 규칙 (필수!)
- 반드시 존댓말을 사용하세요
- "~입니다", "~해요", "~하세요", "~할게요" 체를 사용하세요
- 절대 반말 금지

## 기본 규칙
- 한국어로만 대화하세요
- 이모지 절대 사용 금지
- 짧고 간결하게 말하세요 (최대 2-3문장)
- 항상 "${name}님"으로 호칭하세요

## 숫자 표기 규칙
금액은 반드시 한글로만 말하세요!
- 35,207 → 삼만오천이백칠원
- 아라비아 숫자 절대 금지!

## 역할
- 오늘 지출 현황 안내
- 예산 초과 경고
- 절약 팁 제안
- 지출 패턴 분석
- 재무상담은 하지 않습니다 (상담탭에서 제공)

## ${name}님의 지출 현황
- 이름: ${name} | 나이: ${age}세 | 월수입: ${monthlyIncome}만원
- 일일예산: ${dailyBudget.toLocaleString()}원 | 오늘지출: ${todaySpent.toLocaleString()}원 | 남은예산: ${remainingBudget.toLocaleString()}원
${ragSection}

${name}님의 든든한 지출관리 친구가 되어드릴게요!`;
};

let ClaudeBrain = null;
try {
  ClaudeBrain = require('./agents/ClaudeBrain');
  console.log('[ClaudeBrain] ✅ 로드 완료');
} catch(e) {
  console.error('[ClaudeBrain] ⚠️ 로드 실패:', e.message);
}

let QuestionController = null;
try {
  const qc = require('./agents/QuestionController');
  QuestionController = qc.QuestionController;
  console.log('[QC] ✅ 질문 컨트롤러 로드 완료');
} catch(e) {
  console.error('[QC] ⚠️ 로드 실패:', e.message);
}

let AgentRouter = null;
try {
  AgentRouter = require('./agents/AgentRouter');
  console.log('[AgentRouter] ✅ 멀티에이전트 시스템 로드 완료');
} catch(e) {
  console.error('[AgentRouter] ⚠️ 로드 실패 — 기본 프롬프트 사용:', e.message);
}

const FALLBACK_PROMPT = `당신은 AI재무진단 "머니야"입니다. 오상열 CFP 대표님의 AI 수제자입니다.
공감→복명복창→다음질문 순서로 말합니다. 쌩깝(질문만) 절대 금지.
한국어 존댓말만. 금융상품명 금지. 질문 하나씩.
고객 답변 즉시 update_smart_note 함수 호출.
0=오프닝→1=인적사항(이름/나이/결혼/가족/직업/맞벌이)→2=고민→3=수입지출→4=자산부채→5=집짓기→6=저축투자→7=자산배분→8=종합설계→9=최종의견→10=클로징 순서로 진행.
오원트금융연구소 | AI머니야 | 오상열 CFP`;

const createConsultRealtimePrompt = (userName, financialContext, step = 0, subStep = null, clientData = null) => {
  if (AgentRouter) {
    return AgentRouter.buildPrompt(step, subStep, financialContext?.sessionNo || 1, clientData);
  }
  return FALLBACK_PROMPT;
};

const createSystemPrompt = (userName, financialContext, budgetInfo, ragContext = '') => {
  const name = financialContext?.name || userName || '고객';
  const age = financialContext?.age || 0;
  const monthlyIncome = financialContext?.monthlyIncome || 0;
  const totalAssets = financialContext?.totalAssets || 0;
  const totalDebt = financialContext?.totalDebt || 0;
  const netAssets = financialContext?.netAssets || (totalAssets - totalDebt);
  const wealthIndex = financialContext?.wealthIndex || 0;
  const financialLevel = financialContext?.financialLevel || 0;
  const houseName = financialContext?.houseName || '';
  const livingExpense = financialContext?.livingExpense || 0;
  const savings = financialContext?.savings || 0;
  const pension = financialContext?.pension || 0;
  const insurance = financialContext?.insurance || 0;
  const loanPayment = financialContext?.loanPayment || 0;
  const surplus = financialContext?.surplus || 0;
  const dailyBudget = budgetInfo?.dailyBudget || financialContext?.dailyBudget || 0;
  const todaySpent = budgetInfo?.todaySpent || financialContext?.todaySpent || 0;
  const remainingBudget = budgetInfo?.remainingBudget || financialContext?.remainingBudget || 0;
  const ragSection = ragContext ? `\n## 참고 지식 (RAG)\n${ragContext}\n` : '';

  return `당신은 "머니야"입니다. ${name}님의 개인 AI 금융코치입니다.
오상열 CFP(20년 경력, 2000건 상담)의 금융집짓기® 방법론 전문가입니다.

## 호출 규칙 (최우선!)
- "${name}" 또는 "머니야"라고 부르면: "네, ${name}님!" 이것만 말하고 멈추세요
- 절대 추가 설명하지 마세요
- 그 다음 질문부터 정상 대화하세요

## 첫 응답 규칙 (필수!)
- 고객이 말을 걸면 무조건 "네, ${name}님!" 으로 시작하세요
- 예: "네, ${name}님! 저축률은 월소득의 30% 이상이 좋습니다."
- 절대 "네, ${name}님!" 없이 바로 답변하지 마세요

## 말투 규칙 (필수!)
- 반드시 존댓말을 사용하세요
- "~입니다", "~해요", "~하세요", "~할게요" 체를 사용하세요
- 절대 반말 금지

## 기본 규칙
- 한국어로만 대화하세요
- 이모지 절대 사용 금지
- 짧고 간결하게 말하세요 (최대 2-3문장) — 단, 8단계 상담 진행 시 필요한 설명은 충분히
- 항상 "${name}님"으로 호칭하세요

## 숫자 표기 규칙
금액은 반드시 한글로만 말하세요!
- 35,207 → 삼만오천이백칠원
- 192,000 → 십구만이천원
- 아라비아 숫자 절대 금지!

## 금칙어 (절대 하지 않는 것)
- 특정 상품명·종목명 추천
- 매수·매도 타이밍 판단
- 수익 보장 발언
- "모르겠습니다"로 끝내기 → 항상 다음 질문으로 연결

## ${name}님의 재무 현황
- 이름: ${name} | 나이: ${age}세 | 월수입: ${monthlyIncome}만원
- 총자산: ${totalAssets}만원 | 총부채: ${totalDebt}만원 | 순자산: ${netAssets}만원
- 부자지수: ${wealthIndex}점 | 금융집 레벨: ${financialLevel}단계 (${houseName})
- 생활비: ${livingExpense.toLocaleString()}원 | 저축: ${savings.toLocaleString()}원
- 연금: ${pension.toLocaleString()}원 | 보험: ${insurance.toLocaleString()}원
- 대출상환: ${loanPayment.toLocaleString()}원 | 잉여: ${surplus.toLocaleString()}원
- 일일예산: ${dailyBudget.toLocaleString()}원 | 오늘지출: ${todaySpent.toLocaleString()}원 | 남은예산: ${remainingBudget.toLocaleString()}원
${ragSection}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8단계 상담 리딩 시스템 (핵심)
오상열 CFP 금융집짓기® 방법론 기반
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
당신은 질문을 기다리는 AI가 아닙니다.
먼저 질문하고 A부터 Z까지 리딩합니다.
모든 답변 마지막에는 반드시 다음 질문 또는 다음 단계 안내가 포함됩니다.
공감 먼저 → 분석 → 숫자 → 희망 순서로 말합니다.

[1단계] Opening (5분)
트리거: 고객 첫 메시지 도착 시 즉시 실행
오프닝 멘트:
"반갑습니다! 저는 AI 재무설계사 머니야입니다.
오상열 CFP 선생님의 금융집짓기 방법론으로 고객님의 재무 현황을 함께 살펴드릴게요.
오늘 상담은 수입지출 분석부터 보험, 저축, 투자, 은퇴까지 7대 영역 전체를 60~90분 동안 진행합니다.
먼저 성함과 나이를 알려주시겠어요?"
→ 이름+나이 확인 시 2단계로 자동 전환

[2단계] Fact + Feeling Finding (10분)
트리거: 이름·나이 확인 완료
질문 순서 (하나씩):
① "결혼은 하셨나요? 자녀분은 계신가요? 몇 살인지도 알려주세요."
② "현재 직업은 어떤 일을 하고 계세요? 맞벌이이신가요?"
③ "요즘 돈 관련해서 가장 걱정되시는 게 있으세요?"
④ "재무적으로 가장 이루고 싶은 꿈이 있다면 뭔가요?"
→ 고민 파악 완료 시 3단계로 자동 전환

[3단계] 수입지출·자산부채 분석 (15분)
트리거: 2단계 완료
수입 파악: "가구 합산 월 소득(세후 실수령액)이 얼마나 되세요?"
지출 항목별 (순서대로):
① 생활비 ② 대출 원리금 ③ 보장성 보험료 ④ 노후 연금 납입 ⑤ 저축/투자
부자지수 = (순자산×10) / (나이×월수입×12) × 100
등급: 텐트(0~25%) / 오두막(25~50%) / 빌라(50~100%) / 아파트(100~200%) / 궁전(200%↑)
→ 분석 완료 시 4단계로 자동 전환

[4단계] 금융집짓기 설계도면 소개 (15분)
"집을 한번 그려보시겠습니까? 보통 지붕을 먼저 그리고 기둥을 그리게 됩니다.
하지만 이렇게 지어지는 집은 없습니다. 금융도 똑같아요.
보험이 기초공사, 저축이 기둥, 투자가 지붕입니다."
→ 5단계로 자동 전환

[5단계] 포트폴리오 설계 (15분)
3버킷: 안전(50~60%) / 성장(30~40%) / 꿈(10~20%)
→ 6단계로 자동 전환

[6단계] 7대 영역 종합재무설계 (15분)
순서: 은퇴→부채→저축→투자→세금→부동산→보험
→ 7단계로 자동 전환

[7단계] 최종의견·최종포트폴리오 (10분)
강점 3가지, 개선점 3가지, 금융집 등급
→ 8단계로 자동 전환

[8단계] Closing (5분)
다음 상담 예약, 리포트 발송, 수료증 발급

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
핵심 공식 참고표
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
부자지수    = (순자산×10) / (나이×월수입×12) × 100
DSR(%)      = 월원리금 / 월소득 × 100
저축률(%)   = (저축+연금) / 월소득 × 100
노후필요자금= 월생활비×12×노후생활기간 / 10000 (억원)
보험사망기준= 연봉×3배+총부채
세액공제한도= 연금저축+IRP 합산 연 구백만원
비상예비자금= 월생활비×3~6개월

${name}님의 든든한 금융 친구가 되어드릴게요!`;
};

app.get('/desire.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'desire.html'));
});

app.get('/', (req, res) => {
  if (req.query.mode === 'desire') {
    return res.redirect('/desire.html?mode=beta');
  }
  res.json({
    status: 'AI머니야 서버 실행 중!', version: '9.3 (billing + regular)',
    rag: {
      저서3권: ragData.books.length, AFPK: ragData.afpk.length,
      반퇴시대: ragData.bantoe.length, 명언: ragData.quotes.length,
      문제은행: ragData.questions.length, 워크북: ragData.workbook.length,
      상담사례: ragData.consultation.length, 전문강의: ragData.lecture.length,
      CFHA: ragData.cfha.length, 고객Q: ragData.custQ.length, 잔소리: ragData.nagging.length,
      공식지식베이스: formulaChunks.length,
      total: ragData.books.length + ragData.afpk.length + ragData.bantoe.length +
             ragData.workbook.length + ragData.consultation.length + ragData.lecture.length +
             formulaChunks.length,
    }
  });
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

const multer = require('multer');
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10*1024*1024 } });

app.post('/api/parse-application', upload.single('file'), async (req, res) => {
  try {
    if(!req.file) return res.json({ success: false, error: '파일 없음' });

    const fname = req.file.originalname.toLowerCase();
    let rawText = '';

    const base64 = req.file.buffer.toString('base64');
    const mediaType = fname.endsWith('.xlsx')
      ? 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      : 'application/vnd.ms-excel';

    const Anthropic = require('@anthropic-ai/sdk');
    const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

    const response = await client.messages.create({
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 1000,
      messages: [{
        role: 'user',
        content: [{
          type: 'document',
          source: { type: 'base64', media_type: 'text/plain', data: base64 }
        }, {
          type: 'text',
          text: `이 재무상담 신청서에서 다음 정보를 추출해서 JSON으로만 응답하세요:
{"name":"이름","age":나이숫자,"job":"직업","spouseName":"배우자이름","spouseAge":배우자나이,"spouseJob":"배우자직업","children":"자녀정보","familySize":가족수,"retireAge":은퇴나이,"worry":"경제적고민","dualIncome":true/false,"monthlyIncome":본인월소득만원단위숫자,"spouseIncome":배우자월소득,"totalIncome":합산소득,"totalAsset":총자산만원단위,"totalDebt":총부채만원단위,"netAsset":순자산,"fixedExpense":고정지출,"insurance":보험료,"deposit":예적금}
숫자는 천원단위를 만원단위로 변환하세요. 없는 항목은 0 또는 빈문자열.`
        }]
      }]
    });

    let clientData = {};
    try {
      const txt = response.content[0].text;
      clientData = JSON.parse(txt.replace(/```json|```/g,'').trim());
    } catch(e) {
      clientData = { name:'고객', age:0, monthlyIncome:0 };
    }

    console.log('[신청서] 파싱 완료:', clientData.name, clientData.age+'세');
    res.json({ success: true, clientData });
  } catch(e) {
    console.error('[신청서] 파싱 에러:', e.message);
    res.json({ success: false, error: e.message });
  }
});

const https = require('https');

app.post('/api/desire-feedback', async (req, res) => {
  try {
    const payload = req.body;
    console.log('[DESIRE-FB] 피드백 수신:', JSON.stringify(payload));

    const SHEET_URL = 'https://script.google.com/macros/s/AKfycbwQ0lVhbuSEDLf8E8ILVZbCX2HU1NQgWW-G8yqRXMc3dmRpYbaYUvkBSlCuy9vf9yGTeA/exec';

    const response = await fetch(SHEET_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    console.log('[DESIRE-FB] 구글시트 응답:', response.status);
    res.json({ success: true, message: '피드백 저장 완료' });
  } catch (error) {
    console.error('[DESIRE-FB] 에러:', error.message);
    res.json({ success: false, error: error.message });
  }
});

app.post('/api/rag-search', (req, res) => {
  try {
    const { query, topK = 5 } = req.body;
    const results = searchRAG(query, topK);
    const formulaResults = searchFormulaRAG(query, 2);
    const allResults = [...results, ...formulaResults.map(f => ({ source: '공식', score: 10, topic: f.name, content: f.raw.formula + ' — ' + f.raw.details.slice(0,200) }))];
    res.json({ success: true, query, count: allResults.length, results: allResults });
  } catch (error) { res.json({ success: false, error: error.message }); }
});

app.post('/api/chat', async (req, res) => {
  try {
    const { message, userName, financialContext, budgetInfo } = req.body;
    const ragResults = searchRAG(message, 3);
    const ragContext = ragResults.length > 0 ? ragResults.map(r => `[${r.source}] ${r.topic}: ${r.content}`).join('\n\n') : '';
    const formulaResults = searchFormulaRAG(message, 2);
    const formulaContext = buildFormulaContext(formulaResults);
    const fullRagContext = ragContext + formulaContext;
    const systemPrompt = createSystemPrompt(userName, financialContext, budgetInfo, fullRagContext);
    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: message }],
      max_tokens: 500, temperature: 0.7,
    });
    res.json({ success: true, message: response.choices[0]?.message?.content || '다시 말씀해주세요!' });
  } catch (error) {
    console.error('Chat API Error:', error);
    res.json({ success: false, message: '잠시 후 다시 시도해주세요.' });
  }
});

app.post('/api/tts', async (req, res) => {
  try {
    const { text, voice = 'shimmer' } = req.body;
    const response = await openai.audio.speech.create({ model: 'tts-1', voice, input: text, response_format: 'mp3' });
    const buffer = Buffer.from(await response.arrayBuffer());
    res.json({ success: true, audio: buffer.toString('base64') });
  } catch (error) { res.json({ success: false, error: 'TTS failed' }); }
});

app.post('/api/consult-chat', async (req, res) => {
  try {
    const { message, userName, financialContext, conversationHistory = [] } = req.body;
    const ragResults = searchRAG(message, 3);
    const formulaResults = searchFormulaRAG(message, 2);
    const ragContext = ragResults.map(r => `[${r.source}] ${r.topic}: ${r.content}`).join('\n\n');
    const formulaContext = buildFormulaContext(formulaResults);
    const fullRagContext = ragContext + formulaContext;
    const systemPrompt = createSystemPrompt(userName, financialContext, null, fullRagContext);
    const claudeRes = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 1024,
      system: systemPrompt,
      messages: [
        ...conversationHistory.slice(-10).map(m => ({ role: m.role, content: m.content || m.text })),
        { role: 'user', content: message }
      ]
    });
    const aiText = claudeRes.content[0]?.text || '다시 말씀해주세요.';
    res.json({ success: true, message: aiText, panelData: null, meta: { ragUsed: ragResults.length, formulaUsed: formulaResults.length } });
  } catch (error) {
    console.error('[상담채팅] 에러:', error.status, error.message);
    res.json({ success: false, error: error.message, message: '잠시 후 다시 시도해주세요.' });
  }
});

async function getZoomAccessToken() {
  const credentials = Buffer.from(`${process.env.ZOOM_CLIENT_ID}:${process.env.ZOOM_CLIENT_SECRET}`).toString('base64');
  const params = new URLSearchParams();
  params.append('grant_type', 'account_credentials');
  params.append('account_id', process.env.ZOOM_ACCOUNT_ID);
  const response = await fetch('https://zoom.us/oauth/token', {
    method: 'POST',
    headers: { 'Authorization': `Basic ${credentials}`, 'Content-Type': 'application/x-www-form-urlencoded' },
    body: params.toString()
  });
  const data = await response.json();
  if (!data.access_token) throw new Error(`Zoom 토큰 발급 실패: ${JSON.stringify(data)}`);
  return data.access_token;
}

app.post('/api/zoom/create-meeting', async (req, res) => {
  try {
    const { customerName, scheduledTime, duration = 90 } = req.body;
    const token = await getZoomAccessToken();
    const meetingRes = await fetch('https://api.zoom.us/v2/users/me/meetings', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        topic: `AI머니야 재무상담 — ${customerName}님`, type: scheduledTime ? 2 : 1,
        start_time: scheduledTime || undefined, duration,
        timezone: 'Asia/Seoul',
        settings: { join_before_host: true, mute_upon_entry: false, audio: 'both', auto_recording: 'cloud', waiting_room: false }
      })
    });
    const meeting = await meetingRes.json();
    res.json({ success: true, meetingId: meeting.id, joinUrl: meeting.join_url, startUrl: meeting.start_url, password: meeting.password });
  } catch (error) {
    res.json({ success: false, error: error.message });
  }
});

app.get('/api/zoom/meetings', async (req, res) => {
  try {
    const token = await getZoomAccessToken();
    const response = await fetch('https://api.zoom.us/v2/users/me/meetings', { headers: { 'Authorization': `Bearer ${token}` } });
    const data = await response.json();
    res.json({ success: true, meetings: data.meetings || [] });
  } catch (error) { res.json({ success: false, error: error.message }); }
});

async function elevenLabsTTS(text) {
  const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
  const VOICE_ID = process.env.ELEVENLABS_VOICE_ID;
  const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}/stream`, {
    method: 'POST',
    headers: { 'xi-api-key': ELEVENLABS_API_KEY, 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, model_id: 'eleven_multilingual_v2', voice_settings: { stability: 0.5, similarity_boost: 0.75 }, output_format: 'mp3_44100_128' }),
  });
  if (!response.ok) throw new Error(`ElevenLabs 에러: ${response.status}`);
  return Buffer.from(await response.arrayBuffer());
}

app.post('/api/consult-tts', async (req, res) => {
  try {
    const { text } = req.body;
    const buffer = await elevenLabsTTS(text);
    res.json({ success: true, audio: buffer.toString('base64') });
  } catch (error) {
    try {
      const { text } = req.body;
      const fallback = await openai.audio.speech.create({ model: 'tts-1', voice: 'onyx', input: text, response_format: 'mp3' });
      const buffer = Buffer.from(await fallback.arrayBuffer());
      res.json({ success: true, audio: buffer.toString('base64'), fallback: true });
    } catch { res.json({ success: false, error: 'TTS 실패' }); }
  }
});

const consultRooms = new Map();

app.post('/admin/video-consult/create', async (req, res) => {
  try {
    const { customerName, scheduledTime } = req.body;
    const roomId = `room_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    consultRooms.set(roomId, { host: null, guest: null, customerName, scheduledTime, createdAt: new Date() });
    const joinUrl = `${process.env.FRONTEND_URL || 'https://moneya-develop.vercel.app'}/consult?room=${roomId}`;
    console.log(`[WebRTC] 방 생성: ${roomId} — ${customerName}님`);
    res.json({ success: true, roomId, joinUrl });
  } catch (error) { res.json({ success: false, error: error.message }); }
});

app.get('/video-consult/status/:roomId', (req, res) => {
  const room = consultRooms.get(req.params.roomId);
  if (!room) return res.json({ success: false, error: '존재하지 않는 방' });
  res.json({ success: true, roomId: req.params.roomId, hasHost: !!room.host, hasGuest: !!room.guest, customerName: room.customerName });
});

app.post('/api/generate-report', async (req, res) => {
  try {
    const { noteData, chatHistory } = req.body;
    if (!noteData) {
      return res.status(400).json({ error: '상담 데이터가 없습니다' });
    }

    const prompt = `당신은 오원트금융연구소 AI재무설계사 머니야입니다.
아래 상담 노트 데이터를 바탕으로 종합재무설계 리포트를 JSON으로 작성해주세요.

상담 노트 데이터:
${JSON.stringify(noteData, null, 2)}

${chatHistory ? '대화 요약:\n' + (typeof chatHistory === 'string' ? chatHistory : JSON.stringify(chatHistory).slice(0, 2000)) : ''}

반드시 아래 JSON 구조로만 응답하세요. JSON 외에 다른 텍스트를 포함하지 마세요:
{
  "customer": { "name": "고객명", "age": 0, "job": "직업", "family": 0 },
  "income": { "monthly": 0, "living": 0, "savings": 0, "surplus": 0 },
  "assets": { "total": 0, "financial": 0, "real_estate": 0, "debt": 0, "net": 0 },
  "wealth_index": { "score": 0, "grade": "등급" },
  "desire_stage": { "stage": "단계", "description": "설명" },
  "finance_score": { "total": 0, "expense": 0, "asset": 0, "insurance": 0, "retirement": 0, "debt": 0 },
  "strengths": ["강점1", "강점2", "강점3"],
  "improvements": ["개선점1", "개선점2", "개선점3"],
  "actions": ["액션1", "액션2", "액션3"],
  "next_schedule": "다음 상담 권장일"
}`;

    const claudeRes = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 2000,
      messages: [{ role: 'user', content: prompt }]
    });

    const text = claudeRes.content[0]?.text || '';
    const clean = text.replace(/```json|```/g, '').trim();
    const report = JSON.parse(clean);

    console.log('[DEV-003] 리포트 생성 완료:', report.customer?.name, '재무점수:', report.finance_score?.total);
    res.json(report);

  } catch (error) {
    console.error('[DEV-003] 리포트 생성 오류:', error.message);
    if (error instanceof SyntaxError) {
      res.status(500).json({ error: 'AI 응답 파싱 실패 — 다시 시도해 주세요' });
    } else {
      res.status(500).json({ error: '리포트 생성 실패: ' + error.message });
    }
  }
});

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ★ 페이플 정기결제 자동 청구 API v1.0
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const { google: googleApis } = require('googleapis');

// 페이플 자동 청구 함수
async function paypleRegularCharge({ payerId, amount, goods, orderId, payerName, payerEmail, payerPhone }) {
  // 1단계: 페이플 파트너 인증
  const authRes = await fetch('https://cpay.payple.kr/php/auth.php', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Referer': 'https://financial-house-building.vercel.app' },
    body: JSON.stringify({
      cst_id:  'ohwant',
      custKey: '44cd515b9b59314c0d1ac35653f02ffd1364fa1c4975ddfc526bd079152202d',
    })
  });
  const authData = await authRes.json();
  if (!authData.result || authData.result !== 'success') {
    throw new Error('페이플 인증 실패: ' + JSON.stringify(authData));
  }

  // 2단계: 빌링키로 자동 청구
  const chargeRes = await fetch('https://cpay.payple.kr/php/PayCardAuto.php', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Referer': 'https://financial-house-building.vercel.app',
      'Authorization': 'Basic ' + Buffer.from(authData.AuthKey).toString('base64')
    },
    body: JSON.stringify({
      PCD_CST_ID:       'ohwant',
      PCD_CUST_KEY:     '44cd515b9b59314c0d1ac35653f02ffd1364fa1c4975ddfc526bd079152202d',
      PCD_AUTH_KEY:     authData.AuthKey,
      PCD_PAY_TYPE:     'card',
      PCD_PAY_WORK:     'PAY',
      PCD_PAYER_ID:     payerId,
      PCD_PAYER_NAME:   payerName,
      PCD_PAYER_HP:     payerPhone.replace(/[^0-9]/g, ''),
      PCD_PAYER_EMAIL:  payerEmail,
      PCD_PAY_GOODS:    goods,
      PCD_PAY_TOTAL:    amount,
      PCD_PAY_OID:      orderId,
      PCD_REGULER_FLAG: 'Y',
      PCD_RST_URL:      'https://ohwant-webhook.vercel.app/api/payple',
    })
  });

  const chargeData = await chargeRes.json();
  console.log('[정기청구]', payerName, goods, amount + '원', '->', chargeData.PCD_PAY_RST);
  return chargeData;
}

// 매월 15일 자동 청구 트리거 (make.com 스케줄에서 호출)
// POST /api/billing/charge-all
// Header: x-admin-key: moneya-admin-2026
app.post('/api/billing/charge-all', async (req, res) => {
  const adminKey = req.headers['x-admin-key'];
  if (adminKey !== 'moneya-admin-2026') {
    return res.status(401).json({ error: '인증 실패' });
  }

  try {
    console.log('[정기청구] 자동 청구 시작');

    const saJson = JSON.parse(process.env.GOOGLE_SERVICE_ACCOUNT || '{}');
    const auth   = new googleApis.auth.GoogleAuth({
      credentials: saJson,
      scopes: ['https://www.googleapis.com/auth/spreadsheets'],
    });
    const sheets = googleApis.sheets({ version: 'v4', auth });

    const result = await sheets.spreadsheets.values.get({
      spreadsheetId: process.env.SPREADSHEET_ID,
      range: '빌링키_구독DB!A:J',
    });

    const rows = result.data.values || [];
    if (rows.length <= 1) {
      return res.json({ success: true, message: '청구 대상 없음', charged: 0 });
    }

    const today = new Date();
    const todayDate = today.getDate();

    const subscribers = rows.slice(1).filter(row => {
      return (row[7] || '').trim() === '활성';
    });

    console.log('[정기청구] 활성 구독자:', subscribers.length + '명');

    const results = [];
    for (const row of subscribers) {
      const payerName  = row[1] || '';
      const payerPhone = row[2] || '';
      const payerEmail = row[3] || '';
      const payerId    = row[4] || '';
      const planName   = row[5] || '';
      const amount     = parseInt(row[6] || '0');

      if (!payerId || !amount) continue;

      if (todayDate !== 15) {
        results.push({ payerName, status: 'skipped', reason: '결제일 아님 (오늘:' + todayDate + '일)' });
        continue;
      }

      try {
        const orderId = 'FH_AUTO_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6);
        const chargeResult = await paypleRegularCharge({
          payerId, amount,
          goods: planName + ' 정기결제',
          orderId, payerName, payerEmail, payerPhone
        });

        if (chargeResult.PCD_PAY_RST === 'success') {
          results.push({ payerName, status: 'success', amount, orderId });
          const nextMonth = new Date(today);
          nextMonth.setMonth(nextMonth.getMonth() + 1);
          nextMonth.setDate(15);
          console.log('[정기청구] 성공:', payerName, amount + '원');
        } else {
          results.push({ payerName, status: 'failed', msg: chargeResult.PCD_PAY_MSG });
          console.log('[정기청구] 실패:', payerName, chargeResult.PCD_PAY_MSG);
        }
      } catch (e) {
        results.push({ payerName, status: 'error', error: e.message });
        console.error('[정기청구] 오류:', payerName, e.message);
      }

      await new Promise(r => setTimeout(r, 500));
    }

    const successCount = results.filter(r => r.status === 'success').length;
    const failCount    = results.filter(r => r.status === 'failed').length;
    console.log('[정기청구] 완료 — 성공:', successCount, '실패:', failCount);
    res.json({ success: true, total: results.length, successCount, failCount, results });

  } catch (e) {
    console.error('[정기청구] 전체 오류:', e.message);
    res.status(500).json({ success: false, error: e.message });
  }
});

// 구독 해지 API
// POST /api/billing/cancel
// Body: { payerEmail, reason }
app.post('/api/billing/cancel', async (req, res) => {
  try {
    const { payerEmail, reason } = req.body;
    if (!payerEmail) return res.status(400).json({ error: '이메일 필수' });

    const saJson = JSON.parse(process.env.GOOGLE_SERVICE_ACCOUNT || '{}');
    const auth   = new googleApis.auth.GoogleAuth({
      credentials: saJson,
      scopes: ['https://www.googleapis.com/auth/spreadsheets'],
    });
    const sheets = googleApis.sheets({ version: 'v4', auth });

    const result = await sheets.spreadsheets.values.get({
      spreadsheetId: process.env.SPREADSHEET_ID,
      range: '빌링키_구독DB!A:J',
    });

    const rows = result.data.values || [];
    let foundRowIndex = -1;
    rows.forEach((row, i) => {
      if (i === 0) return;
      if ((row[3] || '').trim().toLowerCase() === payerEmail.toLowerCase()) {
        foundRowIndex = i + 1;
      }
    });

    if (foundRowIndex === -1) {
      return res.status(404).json({ error: '구독 정보를 찾을 수 없습니다' });
    }

    await sheets.spreadsheets.values.update({
      spreadsheetId: process.env.SPREADSHEET_ID,
      range: `빌링키_구독DB!H${foundRowIndex}`,
      valueInputOption: 'USER_ENTERED',
      requestBody: { values: [['해지']] }
    });

    console.log('[구독해지]', payerEmail, '처리 완료. 사유:', reason || '미입력');
    res.json({ success: true, message: '구독이 해지되었습니다. 이번 달 말까지 이용 가능합니다.' });

  } catch (e) {
    console.error('[구독해지] 오류:', e.message);
    res.status(500).json({ success: false, error: e.message });
  }
});
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ★ 페이플 정기결제 API 끝
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const PORT = process.env.PORT || 3001;
const server = app.listen(PORT, () => console.log(`AI머니야 서버 v9.3 시작! 포트: ${PORT}`));

server.keepAliveTimeout = 120000;
server.headersTimeout = 125000;

const wss = new WebSocket.Server({ server });

const activeDesiresessions = new Map();
const activeConsultSessions = new Map();

wss.on('connection', (ws, req) => {
  console.log('[WS] 연결됨');
  const url = new URL(req.url, `http://localhost`);
  const mode = url.searchParams.get('mode');
  console.log(`[WS] 모드: ${mode || 'default'}`);

  const clientIP = req.headers['x-forwarded-for']?.split(',')[0]?.trim() || req.socket.remoteAddress;

  if (mode === 'consult') {
    const existingWs = activeConsultSessions.get(clientIP);
    if (existingWs && existingWs.readyState === 1) {
      console.log(`[상담WS] 중복 세션 차단 — 이전 세션 종료: ${clientIP}`);
      try { existingWs.close(1000, 'duplicate_session'); } catch(e) {}
    }
    activeConsultSessions.set(clientIP, ws);
    ws.on('close', () => {
      if (activeConsultSessions.get(clientIP) === ws) activeConsultSessions.delete(clientIP);
    });
  }

  if (mode === 'desire') {
    const existingWs = activeDesiresessions.get(clientIP);
    if (existingWs && existingWs.readyState === 1) {
      console.log(`[DESIRE-WS] 중복 세션 차단 — IP: ${clientIP}`);
      try { existingWs.close(1000, 'duplicate_session'); } catch(e) {}
    }
    activeDesiresessions.set(clientIP, ws);
    ws.on('close', () => {
      if (activeDesiresessions.get(clientIP) === ws) {
        activeDesiresessions.delete(clientIP);
      }
    });
  }

  let openaiWs = null;
  let userName = '고객';
  let financialContext = null;
  let budgetInfo = null;
  let currentRoomId = null;

  ws.on('message', async (message) => {
    try {
      const msg = JSON.parse(message);

      if (msg.type === 'video_create_room') {
        const roomId = `room_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
        consultRooms.set(roomId, { host: ws, guest: null, createdAt: new Date() });
        currentRoomId = roomId;
        ws.send(JSON.stringify({ type: 'video_room_created', roomId }));
        return;
      }

      if (msg.type === 'video_join_room') {
        const room = consultRooms.get(msg.roomId);
        if (!room) { ws.send(JSON.stringify({ type: 'error', error: '방을 찾을 수 없습니다' })); return; }
        room.guest = ws;
        currentRoomId = msg.roomId;
        if (room.host && room.host.readyState === WebSocket.OPEN) {
          room.host.send(JSON.stringify({ type: 'video_guest_joined' }));
        }
        ws.send(JSON.stringify({ type: 'video_joined', roomId: msg.roomId }));
        return;
      }

      if (msg.type === 'video_signal') {
        const room = consultRooms.get(currentRoomId);
        if (!room) return;
        const target = (ws === room.host) ? room.guest : room.host;
        if (target && target.readyState === WebSocket.OPEN) {
          target.send(JSON.stringify({ type: 'video_signal', signal: msg.signal }));
        }
        return;
      }

      if (msg.type === 'video_end') {
        const room = consultRooms.get(currentRoomId);
        if (room) {
          const other = (ws === room.host) ? room.guest : room.host;
          if (other && other.readyState === WebSocket.OPEN) {
            other.send(JSON.stringify({ type: 'video_ended' }));
          }
          consultRooms.delete(currentRoomId);
        }
        return;
      }

      if (msg.type === 'start_consult' || (msg.type === 'start_app' && mode === 'consult')) {
        console.log('[상담WS] 상담탭 음성 세션 시작');
        userName = msg.userName || '고객';
        financialContext = msg.financialContext || null;

        openaiWs = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17', {
          headers: { 'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`, 'OpenAI-Beta': 'realtime=v1' }
        });

        let collectedData = {};
        let currentConsultStep = 0;

        const qc = QuestionController ? new QuestionController() : null;
        const brain = ClaudeBrain ? new ClaudeBrain(process.env.ANTHROPIC_API_KEY) : null;
        let waitingForAnswer = false;

        let isResponding = false;
        let pendingMessage = null;

        function sendWhenReady(text, delay=200) {
          if (isResponding) {
            pendingMessage = text;
            console.log('[QC] 응답 중 — 큐에 저장');
            return;
          }
          setTimeout(() => {
            if (openaiWs?.readyState === 1) {
              isResponding = true;
              openaiWs.send(JSON.stringify({
                type: 'conversation.item.create',
                item: { type: 'message', role: 'user', content: [{
                  type: 'input_text',
                  text: `[낭독 지시] 다음 문장만 읽으세요. 한 글자도 추가하지 마세요: "${text}"`
                }]}
              }));
              openaiWs.send(JSON.stringify({ type: 'response.create' }));
              console.log(`[QC] 전달: "${text.slice(0,40)}..."`);
            }
          }, delay);
        }

        openaiWs.on('open', () => {
          console.log('[상담WS] OpenAI Realtime 연결 (mini)');
          const name = financialContext?.name || userName || '고객';
          const consultPrompt = createConsultRealtimePrompt(name, financialContext, 0, null, collectedData);
          console.log('[상담WS] 프롬프트 전송 시작 — 길이:', consultPrompt.length, '자');

          openaiWs.send(JSON.stringify({
            type: 'session.update',
            session: {
              modalities: ['text', 'audio'],
              instructions: consultPrompt,
              voice: 'shimmer',
              input_audio_format: 'pcm16',
              output_audio_format: 'pcm16',
              input_audio_transcription: { model: 'whisper-1', language: 'ko' },
              turn_detection: {
                type: 'server_vad',
                threshold: 0.85,
                prefix_padding_ms: 300,
                silence_duration_ms: 1200,
                create_response: true,
              },
              tools: [
                {
                  type: 'function',
                  name: 'update_smart_note',
                  description: '고객이 답변할 때마다 즉시 이 함수를 호출하여 노트에 기록합니다. 텍스트로 설명하지 말고 반드시 함수를 호출하세요. 예: 고객이 "오상열"→ 즉시 호출. 고객이 "55세"→ 즉시 호출. 절대 텍스트로 "(update_smart_note...)"라고 말하지 마세요.',
                  parameters: {
                    type: 'object',
                    properties: {
                      note_page: { type: 'number', description: '0=오프닝 1=인적사항 2=고민 3=수입지출 4=자산부채 5=설계도 6=저축투자 7=자산배분 8=종합설계 9=최종의견 10=클로징' },
                      sub_page:  { type: 'number', description: '8단계 세부(1~7)' },
                      title:     { type: 'string', description: '섹션 제목' },
                      fields:    { type: 'object', description: '실제 값만. 예: {"name":"홍길동"} {"age":"45세"} {"income":"500만원"}. 빈값/플레이스홀더 금지.' }
                    },
                    required: ['note_page', 'title', 'fields']
                  }
                },
                {
                  type: 'function',
                  name: 'clear_smart_note',
                  description: '상담 종료 시 호출',
                  parameters: { type: 'object', properties: { message: { type: 'string' } } }
                }
              ],
              tool_choice: 'auto'
            }
          }));
          ws.send(JSON.stringify({ type: 'session_started' }));
          console.log('[상담WS] session.update 전송 완료 — 프롬프트 적용됨');

          setTimeout(() => {
            if (openaiWs.readyState === 1) {
              const isResume = financialContext?.isResume;
              const currentStep = financialContext?.currentStep || 0;
              const triggerText = isResume
                ? `연결이 잠시 끊어졌다가 재연결됐습니다. 오프닝 없이 ${currentStep}단계부터 바로 이어서 진행하세요.`
                : '오프닝 멘트를 말하고 반드시 멈추세요. 고객이 네 또는 괜찮습니다 라고 할 때까지 기다리세요.';
              openaiWs.send(JSON.stringify({
                type: 'conversation.item.create',
                item: { type: 'message', role: 'user', content: [{ type: 'input_text', text: triggerText }] }
              }));
              openaiWs.send(JSON.stringify({ type: 'response.create' }));
              console.log('[상담WS] 시작 트리거 전송 완료' + (isResume ? ' (재개 모드)' : ''));
            }
          }, 500);
        });

        openaiWs.on('message', (data) => {
          try {
            const event = JSON.parse(data.toString());
            if (event.type === 'response.audio.delta' && event.delta) ws.send(JSON.stringify({ type: 'audio', data: event.delta }));
            if (event.type === 'input_audio_buffer.speech_started') ws.send(JSON.stringify({ type: 'interrupt' }));
            if (event.type === 'response.audio_transcript.done') {
              console.log('[상담WS] 머니야:', event.transcript?.slice(0, 50));
              ws.send(JSON.stringify({ type: 'transcript', text: event.transcript, role: 'assistant' }));
              isResponding = false;
              if (pendingMessage) {
                const msg = pendingMessage;
                pendingMessage = null;
                sendWhenReady(msg, 300);
              }
              if (brain && qc && qc.active && !waitingForAnswer) {
                waitingForAnswer = true;
                console.log('[Brain] 머니야 발화 완료 → 고객 답변 대기 중');
              }

              if (currentConsultStep === 0) {
                currentConsultStep = 1;
                setTimeout(() => {
                  if (openaiWs?.readyState === 1) {
                    const step1Prompt = createConsultRealtimePrompt(
                      financialContext?.name || userName || '고객',
                      financialContext, 1, null, collectedData
                    );
                    openaiWs.send(JSON.stringify({ type: 'session.update', session: { instructions: step1Prompt } }));
                    console.log('[상담WS] → 1단계 프롬프트 주입 완료 (고객 YES 대기 중)');
                  }
                }, 1500);
              }
            }
            if (event.type === 'conversation.item.input_audio_transcription.completed') {
              const userText = (event.transcript || '').trim();
              console.log('[상담WS] 사용자:', userText);
              if (userText.length <= 2) return;

              const sttFix = {
                '노준비':'노후준비','노줄비':'노후준비','노후비':'노후준비',
                '직장애인':'직장인','직장에인':'직장인',
                '오상렬':'오상열','고상렬':'오상열','고상열':'오상열',
                '유료광고':'','본영상은':'','구독좋아요':'',
              };
              let fixedText = userText;
              for(const [w,c] of Object.entries(sttFix)) fixedText = fixedText.split(w).join(c);
              if(fixedText !== userText) console.log('[STT] 보정: "'+userText+'" → "'+fixedText+'"');
              const userTextFinal = fixedText.trim();
              if(userTextFinal.length <= 1) return;

              const noisePatterns = [
                /뉴스/, /기자/, /앵커/, /MBC/, /KBS/, /SBS/, /YTN/, /JTBC/, /TV/, /채널/,
                /안녕하세요$/, /감사합니다$/, /수고하세요$/, /^네\s*네$/, /^고맙습니다$/,
                /이덕영/, /뉴스투데이/, /이브닝뉴스/, /아나운서/,
                /^[ㄱ-ㅎㅏ-ㅣ\s]+$/,
                /^여보세요/, /^잠깐만/, /^뭐라고/, /^다시/, /^취소/,
                /이라고요\?$/, /라고요\?$/, /요\?\s*$/, /^아+$/, /^어+$/,
              ];
              if (noisePatterns.some(p => p.test(userTextFinal))) {
                console.log('[상담WS] STT 소음 차단:', userTextFinal);
                if (openaiWs && openaiWs.readyState === 1) {
                  try { openaiWs.send(JSON.stringify({ type: 'response.cancel' })); } catch(e) {}
                }
                return;
              }
              ws.send(JSON.stringify({ type: 'transcript', text: userText, role: 'user' }));

              if (brain && qc && qc.active && waitingForAnswer) {
                waitingForAnswer = false;
                const currentQ = qc.currentQuestion();
                if (currentQ) {
                  console.log(`[Brain] 고객 발화 처리 중: "${userText}" (질문: ${currentQ.id})`);
                  brain.process(userText, currentQ.id).then(result => {
                    if (result && result.say) {
                      if (result.field && result.value) {
                        Object.assign(collectedData, {[result.field]: result.value});
                        ws.send(JSON.stringify({
                          type: 'smart_note_update',
                          notePage: qc.step,
                          title: '인적사항',
                          fields: {[result.field]: result.value},
                          step: qc.step
                        }));
                        console.log(`[Brain] 노트 저장: ${result.field} = "${result.value}"`);
                      }
                      qc.processAnswer(result.field, result.value, qc.step);
                      waitingForAnswer = true;
                      sendWhenReady(result.say, 300);
                    }
                  }).catch(e => {
                    console.error('[Brain] 처리 오류:', e.message);
                    waitingForAnswer = true;
                  });
                }
              }

              if (qc && !qc.active && currentConsultStep === 1 && !isResponding) {
                const yesPatterns = /^(네|예|괜찮|좋아|응|어|됩니다|좋습니다|알겠|시작|부탁|해주세요)/;
                if (yesPatterns.test(userText.trim())) {
                  qc.activate();
                  const firstQ = qc.currentQuestion();
                  if (firstQ && openaiWs?.readyState === 1) {
                    setTimeout(() => {
                      openaiWs.send(JSON.stringify({
                        type: 'conversation.item.create',
                        item: { type: 'message', role: 'user', content: [{
                          type: 'input_text',
                          text: `[낭독 지시] 다음 문장만 읽으세요. 추가 없이: "감사합니다. 그럼 바로 시작하겠습니다. ${firstQ.ask}"`
                        }]}
                      }));
                      openaiWs.send(JSON.stringify({ type: 'response.create' }));
                      waitingForAnswer = true;
                      console.log(`[QC] 고객 YES 확인 → 첫 질문 전달: "${firstQ.ask}"`);
                    }, 300);
                  }
                }
              }
            }

            if (event.type === 'response.function_call_arguments.done') {
              const fnName = event.name;
              const callId = event.call_id;
              let args = {};
              try { args = JSON.parse(event.arguments || '{}'); } catch(e) {}
              console.log(`[상담FC] ${fnName} 호출:`, JSON.stringify(args));
              let result = '';

              if (fnName === 'search_financial_knowledge') {
                const ragResults = searchRAG(args.query, 5);
                const formulaResults = searchFormulaRAG(args.query, 3);
                const ragText = ragResults.map(r => `[${r.source}] ${r.content}`).join('\n');
                const formulaText = buildFormulaContext(formulaResults);
                const expertMap = { insurance: '보장기준: 사망 연봉3배, 장해3배, 암 연봉1~2배, 뇌1배, 심장1배, 실손5천만원.', retirement: '은퇴4대변수: 은퇴나이(평균73), 수명(90), 월노후생활비(현재70%), 현재준비.', debt_savings: '부채=거실 쓰레기. 신용대출 즉시상환. 비상예비자금=월생활비×6개월.', investment_tax: '기초없이 지붕(투자)만 올리면 무너짐. 골든밸런스7:3.', realestate: '소득에 맞는 크기의 집. 주거비(원리금) 소득30%이하 안전.', budget: '가구원수별: 생활비(1인20%,2인30%,3인40%,4인50%,5인60%), 저축투자 역순.', general: '금융집짓기® 8단계: 지하(보험+비상금)→기둥(부채/저축/은퇴)→지붕(투자/세금)→굴뚝(부동산).' };
                result = `[RAG검색결과]\n${ragText}\n[공식]\n${formulaText}\n[전문지식]\n${expertMap[args.category] || expertMap.general}`;
                ws.send(JSON.stringify({ type: 'note_update', note_type: 'house', highlight: args.category, query: args.query }));
              }
              if (fnName === 'calculate_financial') {
                const inp = args.inputs || {};
                if (args.calculation_type === 'wealth_index') { const index = Math.round((Number(inp.netAssets||0)*10)/(Number(inp.age||30)*Number(inp.monthlyIncome||300)*12)*100); const grade = [[200,'궁전'],[100,'아파트'],[50,'빌라'],[25,'오두막'],[0,'텐트']].find(([m])=>index>=m)?.[1]||'텐트'; result = `부자지수: ${index}점 (${grade}).`; ws.send(JSON.stringify({ type: 'note_update', note_type: 'calc', data: { wealth_index: index, grade } })); }
                else if (args.calculation_type === 'retirement_fund') { const gap = Number(inp.monthlyExpense||250)-Number(inp.publicPension||50)-Number(inp.privatePension||0); const lumpSum = gap*12*(Number(inp.lifeExpectancy||90)-Number(inp.retireAge||65)); result = `월 부족자금: ${gap}만원. 은퇴일시금: ${lumpSum}만원 필요.`; }
                else if (args.calculation_type === 'budget_check') { const income=Number(inp.monthlyIncome||500); const living=Number(inp.livingExpense||0); const family=Number(inp.familySize||1); const stdPct={1:20,2:30,3:40,4:50,5:60}[Math.min(family,5)]||50; const stdAmt=Math.round(income*stdPct/100); const diff=living-stdAmt; result = diff>0?`⚠️ 생활비 ${diff}만원 초과.`:`✅ 생활비 양호.`; }
                else if (args.calculation_type === 'dsr') { const dsr=Math.round((Number(inp.monthlyRepayment||0)*12)/(Number(inp.monthlyIncome||300)*12)*100); result = `DSR: ${dsr}%. ${dsr<=40?'✅ 안전':dsr<=60?'⚠️ 주의':'🚨 위험'}.`; }
                else if (args.calculation_type === 'insurance_gap') { const annual=Number(inp.monthlyIncome||300)*12; result = `사망: ${annual*3}만원, 암: ${annual}~${annual*2}만원, 실손: 5,000만원.`; }
                else { result = '계산 완료.'; }
              }
              if (fnName === 'update_smart_note') {
                const notePage = args.note_page ?? 0;
                const subPage  = args.sub_page  ?? null;
                let fields = {};
                try { fields = typeof args.fields === 'string' ? JSON.parse(args.fields) : (args.fields || {}); } catch(e) { fields = {}; }

                Object.assign(collectedData, fields);

                if (qc && qc.active && Object.keys(fields).length > 0) {
                  const fieldKey = Object.keys(fields)[0];
                  const fieldVal = fields[fieldKey];
                  const qcResult = qc.processAnswer(fieldKey, fieldVal, notePage);

                  if (qcResult && qcResult.text) {
                    if (qcResult.nextQ && qc.step !== currentConsultStep) {
                      currentConsultStep = qc.step;
                      if (openaiWs?.readyState === 1) {
                        const np = createConsultRealtimePrompt(
                          financialContext?.name||userName||'고객',
                          financialContext, qc.step, null, collectedData
                        );
                        openaiWs.send(JSON.stringify({type:'session.update',session:{instructions:np}}));
                        console.log(`[QC] → ${qc.step}단계 프롬프트 전환`);
                      }
                    }
                    sendWhenReady(qcResult.text, 400);
                  }
                }

                function goNextStep(nextStep, nextSubStep, delay) {
                  if (qc && qc.active) {
                    console.log('[상담WS] QC 활성화 중 — goNextStep 건너뜀 (step ' + nextStep + ')');
                    return;
                  }
                  if (!delay) delay = 3500;
                  setTimeout(function() {
                    if (openaiWs && openaiWs.readyState === 1) {
                      try { openaiWs.send(JSON.stringify({type:'response.cancel'})); } catch(e) {}
                      setTimeout(function() {
                        if (!openaiWs || openaiWs.readyState !== 1) return;
                        var np = createConsultRealtimePrompt(
                          (financialContext && financialContext.name) || userName || '고객',
                          financialContext, nextStep, nextSubStep || null, collectedData
                        );
                        openaiWs.send(JSON.stringify({type:'session.update',session:{instructions:np}}));
                        console.log('[상담WS] → ' + nextStep + '단계' + (nextSubStep ? '.' + nextSubStep : '') + ' 전환 완료');
                      }, 600);
                    }
                  }, delay);
                }

                if (notePage===0 && (fields.session||fields.disclaimer)) { goNextStep(1); }
                else if (notePage===1 && fields.dual) { goNextStep(2); }
                else if (notePage===2 && fields.w1) { goNextStep(3); }
                else if (notePage===3 && fields.surplus) { goNextStep(4); }
                else if (notePage===4 && fields.wealth_index) { goNextStep(5); }
                else if (notePage===5 && fields.life_age) { goNextStep(6); }
                else if (notePage===6 && fields.inv_pct) { goNextStep(7); }
                else if (notePage===7 && fields.fin_total) { goNextStep(8, 1); }
                else if (notePage===8) {
                  const sp = subPage||1;
                  if (sp===1 && fields.monthly)   goNextStep(8,2);
                  if (sp===2 && fields.priority)  goNextStep(8,3);
                  if (sp===3 && fields.goal)      goNextStep(8,4);
                  if (sp===4 && fields.rate)      goNextStep(8,5);
                  if (sp===5 && fields.refund)    goNextStep(8,6);
                  if (sp===6 && fields.realty_st) goNextStep(8,7);
                  if (sp===7 && fields.premium)   goNextStep(9);
                }
                else if (notePage===9 && fields.score) { goNextStep(10); }

                ws.send(JSON.stringify({
                  type: 'smart_note_update',
                  notePage, subPage, title: args.title, fields, step: notePage
                }));
                result = `노트${notePage}${subPage ? '-'+subPage : ''} "${args.title}" 기입 완료.`;
              }
              if (fnName === 'clear_smart_note') {
                ws.send(JSON.stringify({ type: 'smart_note_clear', message: args.message||'' }));
                result = '스마트 노트 초기화 완료.';
              }

              openaiWs.send(JSON.stringify({ type: 'conversation.item.create', item: { type: 'function_call_output', call_id: callId, output: result || '처리 완료' } }));
              openaiWs.send(JSON.stringify({ type: 'response.create' }));
            }

            if (event.type === 'error') {
              console.error('[상담WS] OpenAI 에러 상세:', JSON.stringify(event.error, null, 2));
              if (event.error?.code === 'conversation_already_has_active_response') {
                console.log('[상담WS] active_response 충돌 — 자동 취소 처리');
                try { openaiWs.send(JSON.stringify({ type: 'response.cancel' })); } catch(e) {}
              }
              ws.send(JSON.stringify({ type: 'error', error: event.error?.message }));
            }
          } catch (e) { console.error('[상담WS] 메시지 파싱 에러:', e); }
        });

        openaiWs.on('error', (err) => { console.error('[상담WS] OpenAI 에러:', err.message); ws.send(JSON.stringify({ type: 'error', error: err.message })); });
        openaiWs.on('close', () => console.log('[상담WS] OpenAI 연결 종료'));
        return;
      }

      if (msg.type === 'start_desire') {
        console.log('[DESIRE-WS] DESIRE 무료 재무진단 세션 시작');

        const desirePrompt = `당신은 AI 금융집사 "머니야"입니다.
오상열 CFP(20년 경력, 2,000건 상담)의 금융집짓기® 방법론 중 DESIRE 로드맵으로 고객의 재무 현황을 진단합니다.

【절대 원칙】
• 성함을 절대 묻지 마세요.
• 한 번에 질문 하나만 하세요.
• 반드시 존댓말. 한국어로만 대화.
• 이모지·특수문자 사용 금지.
• 금칙어: 재무상담사, 재무설계사, 특정 금융상품 추천, 수익 보장.

【DESIRE 6단계 진단】
오프닝: "안녕하세요! 저는 AI 금융집사 머니야입니다. 오상열 CFP 20년의 노하우로 만들어진 AI 음성 재무진단 서비스예요. 오늘 딱 오분, DESIRE 로드맵으로 고객님의 재무 현재 위치를 정확히 진단해 드릴게요. 개인 재무정보 입력에 동의하시겠어요?"

1단계: 신용대출 여부 확인
2단계: 비상예비자금 확인 (맞벌이 3개월, 외벌이 6개월)
3단계: 가족수 확인 → 저축투자 기준 계산 → 달성 여부
4단계: 금융자산 10억 여부
5단계: 담보대출 여부
6단계: 경제적 자유 단계

클로징: "고객님의 DESIRE 로드맵 진단이 완료되었습니다. 지금 알게 되셨으니 개선하실 수 있어요! 감사합니다!"`;

        openaiWs = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17', {
          headers: { 'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`, 'OpenAI-Beta': 'realtime=v1' }
        });

        openaiWs.on('open', () => {
          console.log('[DESIRE-WS] OpenAI Realtime 연결');
          openaiWs.send(JSON.stringify({
            type: 'session.update',
            session: {
              modalities: ['text', 'audio'],
              instructions: desirePrompt,
              voice: 'shimmer',
              input_audio_format: 'pcm16',
              output_audio_format: 'pcm16',
              input_audio_transcription: { model: 'whisper-1', language: 'ko' },
              turn_detection: { type: 'server_vad', threshold: 0.92, prefix_padding_ms: 500, silence_duration_ms: 2200 },
              tool_choice: 'none'
            }
          }));
          ws.send(JSON.stringify({ type: 'session_started' }));

          setTimeout(() => {
            if (openaiWs && openaiWs.readyState === 1) {
              openaiWs.send(JSON.stringify({
                type: 'conversation.item.create',
                item: { type: 'message', role: 'user', content: [{ type: 'input_text', text: '진단을 시작해주세요. 고객님께 오프닝 멘트로 인사해주세요.' }] }
              }));
              openaiWs.send(JSON.stringify({ type: 'response.create' }));
            }
          }, 500);
        });

        openaiWs.on('message', (data) => {
          try {
            const event = JSON.parse(data.toString());
            if (event.type === 'response.audio.delta' && event.delta) ws.send(JSON.stringify({ type: 'audio', data: event.delta }));
            if (event.type === 'input_audio_buffer.speech_started') ws.send(JSON.stringify({ type: 'interrupt' }));
            if (event.type === 'response.audio_transcript.done') { console.log('[DESIRE-WS] 머니야:', event.transcript?.slice(0, 80)); ws.send(JSON.stringify({ type: 'transcript', text: event.transcript, role: 'assistant' })); }
            if (event.type === 'conversation.item.input_audio_transcription.completed') {
              const userText = (event.transcript || '').trim();
              console.log('[DESIRE-WS] 고객(STT):', userText);
              if (userText.length <= 3) return;
              const noisePatterns = [/뉴스/, /기자/, /앵커/, /MBC/, /KBS/, /SBS/, /YTN/, /안녕하세요$/, /감사합니다$/, /수고하세요$/, /^네\s*네$/];
              if (noisePatterns.some(p => p.test(userText))) return;
              ws.send(JSON.stringify({ type: 'transcript', text: userText, role: 'user' }));
            }
            if (event.type === 'response.done') {
              setTimeout(() => {
                if (ws.readyState === 1) ws.send(JSON.stringify({ type: 'response_done' }));
              }, 1500);
            }
            if (event.type === 'error') { console.error('[DESIRE-WS] OpenAI 에러:', event.error); ws.send(JSON.stringify({ type: 'error', error: event.error?.message })); }
          } catch (e) { console.error('[DESIRE-WS] 메시지 파싱 에러:', e); }
        });

        openaiWs.on('error', (err) => { ws.send(JSON.stringify({ type: 'openai_error', error: err.message })); });
        openaiWs.on('close', (code) => {
          if(code !== 1000 && code !== 1005) {
            try { ws.send(JSON.stringify({ type: 'openai_disconnected', code })); } catch(e) {}
          }
        });
        return;
      }

      if (msg.type === 'start_app') {
        console.log('[Realtime] 앱 시작 요청');
        userName = msg.userName || '고객';
        financialContext = msg.financialContext || null;
        budgetInfo = msg.budgetInfo || null;
        openaiWs = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17', {
          headers: { 'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`, 'OpenAI-Beta': 'realtime=v1' }
        });
        openaiWs.on('open', () => {
          console.log('[Realtime] OpenAI 연결됨!');
          const systemPrompt = createSpendingPrompt(userName, financialContext, budgetInfo);
          openaiWs.send(JSON.stringify({
            type: 'session.update',
            session: {
              modalities: ['text', 'audio'], instructions: systemPrompt, voice: 'shimmer',
              input_audio_format: 'pcm16', output_audio_format: 'pcm16',
              input_audio_transcription: { model: 'whisper-1', language: 'ko' },
              turn_detection: {
                type: 'semantic_vad',
                eagerness: 'low',
                create_response: true,
                interrupt_response: true,
              }
            }
          }));
          ws.send(JSON.stringify({ type: 'session_started' }));
        });
        openaiWs.on('message', (data) => {
          try {
            const event = JSON.parse(data.toString());
            if (event.type === 'response.audio.delta' && event.delta) ws.send(JSON.stringify({ type: 'audio', data: event.delta }));
            if (event.type === 'input_audio_buffer.speech_started') ws.send(JSON.stringify({ type: 'interrupt' }));
            if (event.type === 'response.audio_transcript.done') { console.log('머니야:', event.transcript); ws.send(JSON.stringify({ type: 'transcript', text: event.transcript, role: 'assistant' })); }
            if (event.type === 'conversation.item.input_audio_transcription.completed') { console.log('사용자:', event.transcript); ws.send(JSON.stringify({ type: 'transcript', text: event.transcript, role: 'user' })); }
            if (event.type === 'error') { console.error('OpenAI 에러:', event.error); ws.send(JSON.stringify({ type: 'error', error: event.error?.message })); }
          } catch (e) { console.error('OpenAI 메시지 파싱 에러:', e); }
        });
        openaiWs.on('error', (err) => { ws.send(JSON.stringify({ type: 'error', error: err.message })); });
        openaiWs.on('close', () => console.log('OpenAI 연결 종료'));
      }

      if (msg.type === 'audio' && openaiWs && openaiWs.readyState === WebSocket.OPEN) {
        openaiWs.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: msg.data }));
      }

      if (msg.type === 'ping') {
        ws.send(JSON.stringify({ type: 'pong' }));
        return;
      }

      if (msg.type === 'text_input' && openaiWs && openaiWs.readyState === WebSocket.OPEN) {
        console.log('[상담WS] 텍스트입력:', msg.text);
        ws.send(JSON.stringify({ type: 'transcript', text: msg.text, role: 'user' }));
        openaiWs.send(JSON.stringify({
          type: 'conversation.item.create',
          item: { type: 'message', role: 'user', content: [{ type: 'input_text', text: msg.text }] }
        }));
        openaiWs.send(JSON.stringify({ type: 'response.create' }));
      }

      if (msg.type === 'tts_speak' && msg.text && openaiWs && openaiWs.readyState === WebSocket.OPEN) {
        openaiWs.send(JSON.stringify({
          type: 'conversation.item.create',
          item: { type: 'message', role: 'user', content: [{ type: 'input_text', text: msg.text }] }
        }));
        openaiWs.send(JSON.stringify({ type: 'response.create' }));
      }

      if (msg.type === 'end_session') {
        if (openaiWs) { openaiWs.close(); openaiWs = null; }
      }

      if (msg.type === 'stop') {
        if (openaiWs) openaiWs.close();
      }

    } catch (e) { console.error('메시지 처리 에러:', e); }
  });

  ws.on('close', () => {
    console.log('[WS] 클라이언트 연결 종료');
    if (openaiWs) openaiWs.close();
    if (currentRoomId && consultRooms.has(currentRoomId)) {
      const room = consultRooms.get(currentRoomId);
      const other = (ws === room.host) ? room.guest : room.host;
      if (other && other.readyState === WebSocket.OPEN) {
        other.send(JSON.stringify({ type: 'video_ended' }));
      }
      consultRooms.delete(currentRoomId);
    }
  });
});

console.log('AI머니야 서버 v9.3 초기화 완료!');

const SELF_URL = process.env.RENDER_EXTERNAL_URL || 'https://moneya-server.onrender.com';
setInterval(async () => {
  try {
    await fetch(`${SELF_URL}/api/health`);
    console.log('[핑] 서버 활성 유지 완료');
  } catch(e) {
    console.log('[핑] 실패 (무시):', e.message);
  }
}, 4 * 60 * 1000);
console.log('[핑] 자동 활성 유지 시작 — 콜드 스타트 방지');
