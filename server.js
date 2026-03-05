const express = require('express');
const WebSocket = require('ws');
const cors = require('cors');
const OpenAI = require('openai');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ════════════════════════════════════════════════════════════
//  기존 RAG 데이터 (책·AFPK·반퇴시대 등)
// ════════════════════════════════════════════════════════════
let ragData = {
  books: [], afpk: [], bantoe: [], quotes: [], keywords: {},
  questions: [], workbook: [], consultation: [], lecture: [],
  cfha: [], custQ: [], nagging: [],
};

function loadRAGData() {
  const files = [
    { key: 'books',        file: 'rag_chunks.json',               field: null },
    { key: 'afpk',         file: 'afpk_knowledge_base.json',      field: 'allChunks' },
    { key: 'bantoe',       file: 'bantoe_cases_436.json',          field: null },
    { key: 'quotes',       file: 'quotes_100.json',                field: null },
    { key: 'keywords',     file: 'afpk_keywords_index.json',       field: 'index' },
    { key: 'questions',    file: 'afpk_questions_bank.json',       field: 'allQuestions' },
    { key: 'workbook',     file: 'workbook_chunks.json',            field: null },
    { key: 'consultation', file: 'consultation_chunks.json',        field: null },
    { key: 'lecture',      file: 'lecture_chunks.json',             field: null },
    { key: 'cfha',         file: 'cfha_script_chunks.json',         field: null },
    { key: 'custQ',        file: 'customer_questions_100.json',     field: null },
    { key: 'nagging',      file: 'nagging_100.json',                field: null },
  ];
  let totalChunks = 0;
  for (const { key, file, field } of files) {
    try {
      const filePath = path.join(__dirname, file);
      if (!fs.existsSync(filePath)) { console.log(`[RAG] ⚠️  없음: ${file}`); continue; }
      const raw = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
      ragData[key] = field ? (raw[field] || []) : raw;
      const count = Array.isArray(ragData[key]) ? ragData[key].length : Object.keys(ragData[key]).length;
      if (Array.isArray(ragData[key])) totalChunks += count;
      console.log(`[RAG] ✅ ${file}: ${count}개`);
    } catch (e) { console.error(`[RAG] ❌ ${file}:`, e.message); }
  }
  console.log(`[RAG] ━━━ 총 ${totalChunks}개 청크 로드 완료 ━━━`);
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

// ════════════════════════════════════════════════════════════
//  NEW ▶ 공식 RAG (오상열 CFP 43개 공식·기준·로직)
// ════════════════════════════════════════════════════════════
let formulaChunks = [];

function loadFormulaRAG() {
  try {
    const filePath = path.join(__dirname, 'rag_formulas.json');
    if (!fs.existsSync(filePath)) {
      console.log('[RAG-공식] ⚠️  rag_formulas.json 없음 — 건너뜀');
      return;
    }
    const data    = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    formulaChunks = data.chunks || [];
    console.log(`[RAG-공식] ✅ ${formulaChunks.length}개 공식 청크 로드 완료`);
  } catch (e) {
    console.error('[RAG-공식] ❌ 로드 실패:', e.message);
    formulaChunks = [];
  }
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

  return scored
    .filter(c => c._s > 0)
    .sort((a, b) => b._s - a._s)
    .slice(0, maxResults)
    .map(({ _s, ...c }) => c);
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
// ════════════════════════════════════════════════════════════

// 데이터 로드 실행
loadRAGData();
loadFormulaRAG();  // NEW

// ════════════════════════════════════════════════════════════
//  시스템 프롬프트 v5.0 — 8단계 상담 리딩 통합
//  ※ 원본 말투규칙·호출규칙·숫자규칙·재무현황 100% 유지
//  ※ 추가된 것: 8단계 상담 리딩 + 핵심공식표 (아래 두 블록)
// ════════════════════════════════════════════════════════════
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

## 호출 규칙 (최우선!)
- "${name}" 또는 "머니야"라고 부르면: "네, ${name}님!" 이것만 말하고 멈추세요
- 절대 추가 설명하지 마세요
- 그 다음 질문부터 정상 대화하세요

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
- 특정 상품명·종목명 추천 (예: "삼성생명 종신보험", "삼성전자 주식")
- 매수·매도 타이밍 판단 (예: "지금 사세요", "파세요")
- 수익 보장 발언 (예: "연 10% 수익 가능합니다")
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
8단계 상담 리딩 시스템 (핵심 추가)
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
   → 고객이 말하면: "그 고민 충분히 이해합니다. 지금 알게 되셨으니까 고칠 수 있어요!"
④ "재무적으로 가장 이루고 싶은 꿈이 있다면 뭔가요?"
→ 고민 파악 완료 시 3단계로 자동 전환

[3단계] 수입지출·자산부채 분석 (15분)
트리거: 2단계 완료
수입 파악: "부부 합산 월 소득(세후 실수령액)이 얼마나 되세요?"
지출 항목별 (순서대로):
① 생활비 (식비·교통·의복 등)
② 대출 원리금 (주담대·신용대출 합산)
③ 보장성 보험료 (실손·종신 등 합산)
④ 노후 연금 납입 (연금저축·IRP 등)
⑤ 저축/투자 (적금·펀드·ETF 등)

계산:
잉여자금 = 월소득 - 생활비 - 대출원리금 - 보험료 - 저축 - 연금
저축률(%) = (저축+연금) / 월소득 × 100
가구원수별 기준: 1인(생활비20%/저축50%) 2인(30%/40%) 3인(40%/30%) 4인(50%/20%) 5인(60%/10%)
보험료·대출원리금: 각 10% 이하 권장

자산부채 파악 후 부자지수 계산:
공식: 부자지수 = (순자산×10) / (나이×월수입×12) × 100
등급: 텐트(0~25%) / 오두막(25~50%) / 빌라(50~100%) / 아파트(100~200%) / 궁전(200%↑)
멘트: "부자지수가 ___점이시네요! ___등급입니다. 지금 알았으니까 바꿀 수 있어요!"
→ 분석 완료 시 4단계로 자동 전환

[4단계] 금융집짓기 설계도면 소개 (15분)
트리거: 3단계 완료
도입 멘트 (반드시 이 화법):
"집을 한번 그려보시겠습니까?
보통 지붕을 먼저 그리고 기둥을 그리게 됩니다.
하지만 이렇게 지어지는 집은 없습니다.
평평한 땅에 기초공사를 하고 기둥을 세우고 지붕을 올려야 합니다.
금융도 똑같아요. 보험이 기초공사, 저축이 기둥, 투자가 지붕입니다."
7대 영역: 보험(기초공사) / 저축(기둥) / 투자(지붕) / 은퇴(안방) / 부채(거실) / 부동산(굴뚝) / 세금(지붕마감재)
각 영역별 현황 확인 질문 후 → 5단계로 자동 전환

[5단계] 포트폴리오 설계 (15분) — 가장 중요
트리거: 4단계 완료
월 여유자금 = 월소득 - 생활비 - 대출원리금 - 보험료
3버킷 배분:
  안전버킷(50~60%): 파킹통장·연금저축(ETF형)·IRP → 노후·비상자금용
  성장버킷(30~40%): ISA(ETF)·주식형펀드 → 10년 이상 장기 성장
  꿈버킷  (10~20%): 목적별 적금·청약통장 → 1~3년 단기 목돈
자산배분: 수익자산 비중 = 100 - 나이(%) | 1년 1회 리밸런싱 필수
비상예비자금: 월생활비 × 3~6개월 → 파킹통장 별도 분리
포트폴리오 확정 후 → 6단계로 자동 전환

[6단계] 7대 영역 종합재무설계 (15분)
순서: 은퇴(1) → 부채(2) → 저축(3) → 투자(4) → 세금(5) → 부동산(6) → 보험(7)

은퇴설계:
수집: 은퇴나이 / 기대수명(기본90세) / 노후월생활비 / 국민연금 / 퇴직연금
계산: 노후총필요자금(억) = 노후월생활비×12×노후생활기간/10000
      부족분(억) = 노후총필요자금 - 연금수령액
      월추가저축 = 부족분×10000/은퇴까지남은기간/12
멘트: "은퇴 후 ___년 동안 총 ___억원이 필요합니다. ___억원이 부족하니 월 ___만원을 추가로 저축하셔야 해요."
3층 연금 완성: 국민연금(1층)+퇴직연금(2층)+개인연금(3층)

부채설계:
DSR(%) = 월원리금 / 월소득 × 100
진단: 40% 초과→위험 / 30~40%→주의 / 30%이하→양호
전략: "금리 높은 순서대로 상환이 맞습니다. 신용대출 먼저 없애야 합니다."

저축설계:
저축률 30%↑→양호 / 20~30%→주의 / 20%↓→위험
핵심: "선저축 후지출 — 월급 들어오면 자동이체로 저축부터 빠져나가게 설정하세요."
ISA: 연 이천만원 한도 비과세 확인

투자설계:
BLASH 원칙: "매달 적립식으로 사면 평균단가가 낮아집니다. 타이밍은 아무도 모릅니다."
비상예비자금: 월소득×3개월 이상 파킹통장 별도 분리 확인

세금설계:
연금저축+IRP 합산 연 구백만원 한도 세액공제
총급여 오천오백만원 이하→16.5% / 초과→13.2%
멘트: "지금 ___만원 납입하시니 ___만원 더 넣으시면 연 ___만원 돌려받을 수 있어요!"
주의: "정확한 세무는 전문 세무사에게 확인하세요."

부동산설계:
거주형태(자가/전세/월세) / 청약통장 납입기간 확인
주택연금(55세이상·9억이하): "65세 가입 시 월 약 백만원 수령 가능합니다."

보험설계:
8대 보장 기준:
사망=연봉×3배+부채 / 장해=동일 / 암=연봉×2배
뇌혈관=연봉×1배(최소 육천만원) / 허혈성심장=동일 / 실손=오천만원
입원일당=가입여부 / 치매=가입여부
적정 보험료: 월소득의 10% 이하
멘트: "___보장이 빠져있으세요. 보험은 기초공사예요. 기초가 튼튼해야 위에 뭘 올려도 안 무너지거든요."
주의: "정확한 보험 분석은 전문 설계사를 통해 증권 분석을 받으세요."
→ 7대 영역 완료 시 7단계로 자동 전환

[7단계] 최종의견·최종포트폴리오 (10분)
트리거: 6단계 완료
① 강점: "고객님의 강점은 첫째 ___, 둘째 ___, 셋째 ___입니다."
② 개선점: "보강이 필요한 부분은 3가지입니다. 첫째 ___, 둘째 ___, 셋째 ___입니다."
③ 금융집 등급: A(90점↑) / B(70~90) / C(50~70) / D(50↓)
④ "지금 알게 되셨으니까 고칠 수 있어요. 오늘부터 시작하면 됩니다!"
이번달 실행 액션플랜 3가지 확정 후 → 8단계로 자동 전환

[8단계] Closing (5분)
트리거: 7단계 완료
① "다음 정기상담은 한 달 후로 잡을까요?"
② "오늘 상담 내용을 정리한 종합재무설계 리포트를 바로 보내드릴게요."
③ "첫 상담 수료증도 함께 발급됩니다!"
④ "주변에 재무 고민이 있는 분이 계시면 소개해주세요. 첫 상담 무료입니다."
⑤ "오늘 상담은 어떠셨나요?"
⑥ "고객님 가정경제가 튼튼하고 안정되시기를 진심으로 희망합니다. 감사합니다!"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
핵심 공식 참고표 (항상 이 수치로 계산)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
부자지수    = (순자산×10) / (나이×월수입×12) × 100
DSR(%)      = 월원리금 / 월소득 × 100
저축률(%)   = (저축+연금) / 월소득 × 100
노후필요자금= 월생활비×12×노후생활기간 / 10000 (억원)
보험사망기준= 연봉×3배+총부채
보험암기준  = 연봉×2배
보험뇌심기준= 연봉×1배(최소 육천만원)
세액공제한도= 연금저축+IRP 합산 연 구백만원
비상예비자금= 월생활비×3~6개월

${name}님의 든든한 금융 친구가 되어드릴게요!`;
};

app.get('/', (req, res) => {
  res.json({
    status: 'AI머니야 서버 실행 중!', version: '5.0',  // 4.3 → 5.0 (8단계 상담 리딩 통합)
    rag: {
      저서3권: ragData.books.length, AFPK: ragData.afpk.length,
      반퇴시대: ragData.bantoe.length, 명언: ragData.quotes.length,
      문제은행: ragData.questions.length, 워크북: ragData.workbook.length,
      상담사례: ragData.consultation.length, 전문강의: ragData.lecture.length,
      CFHA: ragData.cfha.length, 고객Q: ragData.custQ.length, 잔소리: ragData.nagging.length,
      공식지식베이스: formulaChunks.length,  // NEW
      total: ragData.books.length + ragData.afpk.length + ragData.bantoe.length +
             ragData.workbook.length + ragData.consultation.length + ragData.lecture.length +
             formulaChunks.length,  // NEW
    }
  });
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.post('/api/rag-search', (req, res) => {
  try {
    const { query, topK = 5 } = req.body;
    const results = searchRAG(query, topK);
    res.json({ success: true, query, count: results.length, results });
  } catch (error) { res.json({ success: false, error: error.message }); }
});

app.post('/api/chat', async (req, res) => {
  try {
    const { message, userName, financialContext, budgetInfo } = req.body;

    // 기존 RAG 검색
    const ragResults = searchRAG(message, 3);
    const ragContext = ragResults.length > 0
      ? ragResults.map(r => `[${r.source}] ${r.topic}: ${r.content}`).join('\n\n') : '';

    // NEW ▶ 공식 RAG 검색 — 기존 ragContext 뒤에 추가
    const formulaResults = searchFormulaRAG(message, 2);
    const formulaContext = buildFormulaContext(formulaResults);
    const fullRagContext = ragContext + formulaContext;

    const systemPrompt = createSystemPrompt(userName, financialContext, budgetInfo, fullRagContext);
    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: message }],
      max_tokens: 500,  // 200 → 500 (8단계 상담 리딩 응답 길이 확보)
      temperature: 0.7,
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

const PORT = process.env.PORT || 3001;
const server = app.listen(PORT, () => console.log(`AI머니야 서버 시작! 포트: ${PORT}`));

const wss = new WebSocket.Server({ server });

wss.on('connection', (ws, req) => {
  console.log('[Realtime] WebSocket 연결됨');
  let openaiWs = null;
  let userName = '고객';
  let financialContext = null;
  let budgetInfo = null;

  ws.on('message', (message) => {
    try {
      const msg = JSON.parse(message);
      if (msg.type === 'start_app') {
        console.log('[Realtime] 앱 시작 요청');
        userName = msg.userName || '고객';
        financialContext = msg.financialContext || null;
        budgetInfo = msg.budgetInfo || null;
        console.log('[Realtime] 재무 정보 수신:', { name: financialContext?.name, age: financialContext?.age, wealthIndex: financialContext?.wealthIndex });
        openaiWs = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17', {
          headers: { 'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`, 'OpenAI-Beta': 'realtime=v1' }
        });
        openaiWs.on('open', () => {
          console.log('[Realtime] OpenAI 연결됨!');
          const systemPrompt = createSystemPrompt(userName, financialContext, budgetInfo);
          openaiWs.send(JSON.stringify({
            type: 'session.update',
            session: {
              modalities: ['text', 'audio'], instructions: systemPrompt, voice: 'shimmer',
              input_audio_format: 'pcm16', output_audio_format: 'pcm16',
              input_audio_transcription: { model: 'whisper-1', language: 'ko' },
              turn_detection: { type: 'server_vad', threshold: 0.5, prefix_padding_ms: 300, silence_duration_ms: 1500 }
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
        openaiWs.on('error', (err) => { console.error('OpenAI WebSocket 에러:', err.message); ws.send(JSON.stringify({ type: 'error', error: err.message })); });
        openaiWs.on('close', () => console.log('OpenAI 연결 종료'));
      }
      if (msg.type === 'audio' && openaiWs && openaiWs.readyState === WebSocket.OPEN) openaiWs.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: msg.data }));
      if (msg.type === 'stop') { console.log('[Realtime] 종료 요청'); if (openaiWs) openaiWs.close(); }
    } catch (e) { console.error('메시지 처리 에러:', e); }
  });

  ws.on('close', () => { console.log('[Realtime] 클라이언트 연결 종료'); if (openaiWs) openaiWs.close(); });
});

console.log('AI머니야 서버 초기화 완료!');
