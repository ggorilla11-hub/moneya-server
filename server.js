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

// ════════════════════════════════════════════════════════════
//  AI 클라이언트 (OpenAI — TTS/음성, Anthropic — 텍스트 채팅)
// ════════════════════════════════════════════════════════════
const openai    = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// ════════════════════════════════════════════════════════════
//  기존 RAG 데이터 (책·AFPK·반퇴시대 등) — 100% 유지
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
//  공식 RAG (오상열 CFP 43개) — v4.3에서 이어받음
// ════════════════════════════════════════════════════════════
let formulaChunks = [];

function loadFormulaRAG() {
  try {
    const filePath = path.join(__dirname, 'rag_formulas.json');
    if (!fs.existsSync(filePath)) { console.log('[RAG-공식] ⚠️  없음'); return; }
    const data    = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    formulaChunks = data.chunks || [];
    console.log(`[RAG-공식] ✅ ${formulaChunks.length}개 청크 로드 완료`);
  } catch (e) { console.error('[RAG-공식] ❌:', e.message); formulaChunks = []; }
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

// ════════════════════════════════════════════════════════════
//  수석 머니야 시스템 프롬프트
// ════════════════════════════════════════════════════════════
const createSystemPrompt = (userName, financialContext, budgetInfo, ragContext = '') => {
  const name            = financialContext?.name || userName || '고객';
  const age             = financialContext?.age || 0;
  const monthlyIncome   = financialContext?.monthlyIncome || 0;
  const totalAssets     = financialContext?.totalAssets || 0;
  const totalDebt       = financialContext?.totalDebt || 0;
  const netAssets       = financialContext?.netAssets || (totalAssets - totalDebt);
  const wealthIndex     = financialContext?.wealthIndex || 0;
  const financialLevel  = financialContext?.financialLevel || 0;
  const houseName       = financialContext?.houseName || '';
  const livingExpense   = financialContext?.livingExpense || 0;
  const savings         = financialContext?.savings || 0;
  const pension         = financialContext?.pension || 0;
  const insurance       = financialContext?.insurance || 0;
  const loanPayment     = financialContext?.loanPayment || 0;
  const surplus         = financialContext?.surplus || 0;
  const dailyBudget     = budgetInfo?.dailyBudget || financialContext?.dailyBudget || 0;
  const todaySpent      = budgetInfo?.todaySpent || financialContext?.todaySpent || 0;
  const remainingBudget = budgetInfo?.remainingBudget || financialContext?.remainingBudget || 0;
  const ragSection      = ragContext ? `\n## 참고 지식 (RAG)\n${ragContext}\n` : '';

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
- 짧고 간결하게 말하세요 (최대 2-3문장)
- 항상 "${name}님"으로 호칭하세요

## 숫자 표기 규칙
금액은 반드시 한글로만 말하세요!
- 35,207 → 삼만오천이백칠원
- 192,000 → 십구만이천원
- 아라비아 숫자 절대 금지!

## ${name}님의 재무 현황
- 이름: ${name} | 나이: ${age}세 | 월수입: ${monthlyIncome}만원
- 총자산: ${totalAssets}만원 | 총부채: ${totalDebt}만원 | 순자산: ${netAssets}만원
- 부자지수: ${wealthIndex}점 | 금융집 레벨: ${financialLevel}단계 (${houseName})
- 생활비: ${livingExpense.toLocaleString()}원 | 저축: ${savings.toLocaleString()}원
- 연금: ${pension.toLocaleString()}원 | 보험: ${insurance.toLocaleString()}원
- 대출상환: ${loanPayment.toLocaleString()}원 | 잉여: ${surplus.toLocaleString()}원
- 일일예산: ${dailyBudget.toLocaleString()}원 | 오늘지출: ${todaySpent.toLocaleString()}원 | 남은예산: ${remainingBudget.toLocaleString()}원
${ragSection}
${name}님의 든든한 금융 친구가 되어드릴게요!`;
};

// ════════════════════════════════════════════════════════════
//  멀티에이전트 시스템 — 에이전트 프롬프트
// ════════════════════════════════════════════════════════════
const AGENT_PROMPTS = {

  orchestrator: `당신은 오케스트레이터입니다. 고객 메시지를 분석하여 어떤 전문 에이전트가 필요한지 판단합니다.
응답은 반드시 JSON으로만 하세요. 설명 없이 JSON만:
{"agents": ["에이전트1", "에이전트2"], "priority": "가장 중요한 에이전트", "reasoning": "판단 이유"}

가능한 에이전트: memory, analysis, statistics, insurance, retirement, debt_savings, investment_tax, realestate, emotion, compliance

규칙:
- memory는 항상 포함
- 감정적 표현(불안, 화남, 걱정, 힘들어)이 있으면 emotion을 최우선으로
- 금칙어 유도(상품추천, 수익보장 등)가 감지되면 compliance를 최우선으로
- 구체적 수치 계산이 필요하면 analysis 포함
- 통계/비교가 필요하면 statistics 포함
- 최소 2개, 최대 4개 에이전트만 호출 (비용 최적화)`,

  memory: `당신은 기억 머니야입니다. 고객의 과거 대화 맥락을 관리합니다.
응답은 반드시 JSON으로만 하세요:
{"relevantHistory": "관련 과거 맥락 요약", "activeGoals": ["목표 상태"], "customerPersonality": "선호 스타일", "newInfoToSave": "새로 알게 된 정보"}`,

  analysis: `당신은 분석 머니야입니다. 정확한 수치 계산만 담당합니다.

예산 기준표 (반드시 이 기준으로 판단):
| 항목 | 1인 | 2인 | 3인 | 4인 | 5인 |
|------|-----|-----|-----|-----|-----|
| 생활비 | 20% | 30% | 40% | 50% | 60% |
| 저축투자 | 50% | 40% | 30% | 20% | 10% |
| 노후연금 | 10% | 10% | 10% | 10% | 10% |
| 보장성보험 | 10% | 10% | 10% | 10% | 10% |
| 대출원리금 | 10% | 10% | 10% | 10% | 10% |

응답은 반드시 JSON으로만 하세요:
{"calculations": [{"name": "항목명", "formula": "계산식", "result": "결과", "korean": "한글표현"}], "budgetDiagnosis": {"status": "양호/초과/부족", "details": "상세내용"}, "fhbScore": {"total": 0, "breakdown": {}}}`,

  statistics: `당신은 통계 머니야입니다. 한국 금융 통계와 비교 데이터를 제공합니다.

한국 주요 통계:
- 가계 평균 저축률: 약 35% (2인 이상 가구)
- 국민연금 평균 수령액: 약 52만원/월
- 가계 평균 부채: 약 9,170만원
- 평균 보험료: 소득의 약 12%
- 평균 은퇴 나이: 55세(실질), 65세(국민연금)
- 노후 필요 생활비: 월 200~300만원 (통계청)

응답은 반드시 JSON으로만 하세요:
{"comparison": "동일 연령/소득 대비 고객 위치", "benchmark": "해당 항목의 한국 평균", "similarCases": "유사 사례 인사이트", "insight": "수석 머니야에게 전달할 핵심 정보"}`,

  insurance: `당신은 보험설계 머니야입니다. 금융집짓기의 지하(기초공사)를 담당합니다.

보장 적정 기준:
- 사망: 연봉의 3배
- 장해: 연봉의 3배
- 암진단비: 연봉의 1~2배
- 뇌혈관: 연봉의 1배
- 심장: 연봉의 1배
- 실손: 5천만원
보험료 기준: 소득의 10%

분석 결과 제공. 구체적 상품명은 절대 언급 금지.
응답은 반드시 JSON으로만 하세요:
{"coverageGap": [{"type": "사망", "current": "X원", "ideal": "Y원", "gap": "부족/충분"}], "premiumAnalysis": {"monthly": "X원", "ratio": "소득의 X%", "verdict": "적정/과다/부족"}, "recommendations": ["방향 제안"], "needExpert": false, "fhbBasementScore": 0}`,

  retirement: `당신은 은퇴설계 머니야입니다. 금융집짓기의 안방을 담당합니다.
안방은 인생에서 제일 중요한 방입니다.

은퇴 4대 변수: 은퇴나이(기본 73세), 예상수명(기본 90세), 월노후생활비(기본 현재 생활비 70%), 현재준비상태

계산공식:
1단계: 월 부족자금 = 노후생활비 - 공적연금 - 개인연금
2단계: 은퇴일시금 = 월 부족자금 × 12 × (수명 - 은퇴나이)
3단계: 순은퇴일시금 = 은퇴일시금 - 퇴직연금
4단계: 월 필요 납입액 = 순은퇴일시금 ÷ (은퇴나이 - 현재나이) ÷ 12

응답은 반드시 JSON으로만 하세요:
{"variables": {"retireAge": 0, "lifeExpectancy": 0, "monthlyNeeded": "X원", "currentPrep": "X원"}, "calculation": {"gap": "월 X원 부족", "lumpSum": "X원 필요", "monthlySaving": "월 X원 저축 필요"}, "pensionLayers": {"public": "X원", "corporate": "X원", "private": "X원"}, "fireTarget": "목돈 X원 → 월 X원 연금", "urgency": "시급/보통/여유", "fhbPillarScore": 0}`,

  debt_savings: `당신은 부채/저축 머니야입니다. 거실(부채)과 건넌방(저축)을 담당합니다.

부채 상환 우선순위:
- 신용대출: 즉시, 금액 작은 것부터 (행동경제학)
- 담보대출: 은퇴 시까지 상환

비상예비자금: 월 생활비 × 6개월 이상

응답은 반드시 JSON으로만 하세요:
{"debtPriority": [{"type": "신용대출", "amount": "X원", "action": "즉시 상환"}], "emergencyFund": {"current": "X원", "ideal": "X원", "status": "충분/부족"}, "savingsDesign": {"purpose": "목적", "period": "기간", "monthly": "X원"}, "fhbScore": {"livingRoom": 0, "guestRoom": 0}}`,

  investment_tax: `당신은 투자/세금 머니야입니다. 다락방(투자)과 세금을 담당합니다.

투자 원칙 (금융집짓기):
- 기초(보험)+기둥(저축) 완성 후 투자 시작
- 골든밸런스 7:3 (안전자산 70% : 위험자산 30%)
- 연 1~2회 리밸런싱 (BLASH 원칙)
- ETF 분산투자 권장

절세 수단: 연금저축(400만원), IRP(300만원), ISA(2,000만원)

응답은 반드시 JSON으로만 하세요:
{"readiness": "투자 준비도", "allocation": {"safe": "X%", "risk": "X%"}, "taxSaving": {"available": ["항목"], "annualBenefit": "X원"}, "recommendation": "방향 제안", "fhbRoofScore": 0}`,

  realestate: `당신은 부동산 머니야입니다. 굴뚝(부동산)을 담당합니다.

부동산 기준:
- 주거용 1채: 필수 (굴뚝)
- LTV: 규제지역 50%, 비규제 70%
- DTI: 규제지역 40%, 비규제 50%
- DSR: 전국 40%

응답은 반드시 JSON으로만 하세요:
{"homeStatus": "자가/전세/월세", "purchasePlan": "현황 분석", "loanLimit": {"ltv": "X원", "dsr": "X원"}, "timeline": "X년 후 매매 가능 예상", "fhbChimneyScore": 0}`,

  emotion: `당신은 감정 머니야입니다. 고객의 감정을 읽는 전문가입니다.

감정 분류:
- 불안/걱정 → 공감 먼저, 숫자로 안심
- 화남/분노 → 방어하지 말고, 인정+공감+대안
- 자신감 과잉 → 칭찬하되 리스크 언급
- 슬픔/좌절 → 감정 수용, 조언은 나중에
- 불신/경계 → 강요 않고 전문가 연결 안내
- 무관심 → 관심 끌 수 있는 숫자/질문 제시

위기 즉시 플래그:
- 극단적 선택 암시 → 전문 기관 안내
- 사기 피해 의심 → 금감원 1332

응답은 반드시 JSON으로만 하세요:
{"currentEmotion": "감정 상태", "emotionChange": "변화", "suggestedTone": "수석이 사용할 톤", "warningFlags": [], "coupleConflict": false}`,

  compliance: `당신은 안전 머니야입니다. 모든 답변의 최종 검수를 담당합니다.

1차 금칙어 (즉시 차단):
- 특정 상품명 추천, 수익/원금 보장, 매수/매도 지시, 탈세 조언

2차 금칙어 (자동 대체):
- "이 상품이 좋습니다" → "이런 유형을 알아보시면"
- "가입하세요" → "전문가와 상의해보세요"
- "무조건" → "일반적으로"

Shadow Mode 채점 (각 10점, 총 70점):
1. 정확성 2. 적절성 3. 순서 준수 4. 공감도 5. 안전성 6. 못과 액자 7. 한계 인정
등급: S(63+) A(56+) B(49+) C(42+) F(42미만)

응답은 반드시 JSON으로만 하세요:
{"approved": true, "violations": [], "corrections": "", "shadowScore": {"total": 70, "grade": "S"}, "improvements": ""}`,
};

// ════════════════════════════════════════════════════════════
//  멀티에이전트 핵심 함수
// ════════════════════════════════════════════════════════════

// 간단한 인사/잡담 감지 (에이전트 스킵 → 비용 최적화)
const SIMPLE_PATTERNS = ['안녕', '반가워', '고마워', '감사', '수고', '잘가', '또봐'];
function isSimpleMessage(msg) {
  return SIMPLE_PATTERNS.some(p => msg.includes(p)) && msg.length < 20;
}

// 단일 에이전트 실행
async function runAgent(agentName, message, customerData) {
  const prompt = AGENT_PROMPTS[agentName];
  const contextStr = JSON.stringify({
    financialData: customerData.financialContext || {},
    customerSummary: customerData.summary || {},
    recentMessages: customerData.recentMessages || [],
    insuranceData: customerData.insurance || {},
    retirementData: customerData.retirement || {},
    debtSavingsData: customerData.debtSavings || {},
    investmentData: customerData.investment || {},
    realEstateData: customerData.realEstate || {},
    currentConversation: customerData.conversation || [],
    draftResponse: customerData.draftResponse || '',
    name: customerData.name || '고객',
  });

  const response = await anthropic.messages.create({
    model: 'claude-sonnet-4-5',
    max_tokens: 500,
    system: prompt + `\n\n[고객 데이터]\n${contextStr}`,
    messages: [{ role: 'user', content: message }],
  });

  try {
    const text = response.content[0].text.trim();
    // JSON 파싱 (코드블록 제거 후)
    const clean = text.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
    return { agent: agentName, result: JSON.parse(clean) };
  } catch {
    return { agent: agentName, result: { raw: response.content[0].text } };
  }
}

// 멀티에이전트 전체 오케스트레이션
async function multiAgentChat(message, customerData, conversationHistory) {

  // 1단계: 오케스트레이터 — 필요 에이전트 결정
  const routing     = await runAgent('orchestrator', message, customerData);
  const agentsToRun = (routing.result.agents || ['memory', 'analysis']).filter(a => a !== 'compliance');
  console.log(`[멀티에이전트] 라우팅: ${agentsToRun.join(', ')} (우선: ${routing.result.priority || '-'})`);

  // 2단계: 전문 에이전트 병렬 실행 (compliance 제외 — 나중에 검수)
  const agentResults = await Promise.all(
    agentsToRun.map(agent => runAgent(agent, message, customerData))
  );

  // 3단계: RAG 검색 (기존 + 공식)
  const ragResults     = searchRAG(message, 3);
  const formulaResults = searchFormulaRAG(message, 2);
  const formulaCtx     = buildFormulaContext(formulaResults);

  const ragContext = [
    ragResults.map(r => `[${r.source}] ${r.topic}: ${r.content}`).join('\n\n'),
    formulaCtx,
  ].filter(Boolean).join('\n');

  // 4단계: 에이전트 분석 결과 종합
  const agentContext = agentResults
    .map(r => `[${r.agent} 분석]\n${JSON.stringify(r.result, null, 2)}`)
    .join('\n\n');

  // 5단계: 수석 머니야가 종합 답변 생성
  const chiefSystemPrompt = createSystemPrompt(
    customerData.name,
    customerData.financialContext,
    customerData.budgetInfo,
    ragContext
  ) + `

[전문 에이전트 분석 결과 — 고객에게 보이지 않음, 자연스럽게 녹여서 활용]
${agentContext}

★ 중요: "분석 결과에 따르면" 같은 표현 금지. 당신이 직접 분석한 것처럼 말하세요.
★ 금액은 반드시 한글로만 (아라비아 숫자 금지).
★ 최대 2~3문장으로 간결하게.`;

  const chiefResponse = await anthropic.messages.create({
    model: 'claude-sonnet-4-5',
    max_tokens: 400,
    system: chiefSystemPrompt,
    messages: [
      ...conversationHistory.map(m => ({ role: m.role, content: m.content })),
      { role: 'user', content: message },
    ],
  });

  const draftAnswer = chiefResponse.content[0].text;

  // 6단계: 안전 머니야 최종 검수 (금칙어 감지된 경우에만)
  const dangerWords = ['추천해', '가입하세', '보장', '무조건', '삼성', '한화', '교보'];
  const needCompliance = dangerWords.some(w => draftAnswer.includes(w));

  let finalAnswer    = draftAnswer;
  let shadowScore    = null;

  if (needCompliance) {
    const compliance = await runAgent('compliance', draftAnswer, {
      ...customerData, draftResponse: draftAnswer,
    });
    finalAnswer  = compliance.result.approved
      ? draftAnswer
      : (compliance.result.corrections || draftAnswer);
    shadowScore  = compliance.result.shadowScore;
    console.log(`[안전 머니야] 검수 완료: ${compliance.result.approved ? '승인' : '수정'} | 등급: ${shadowScore?.grade}`);
  }

  // 7단계: 기억 머니야 — 비동기 요약 저장 (응답 후)
  setImmediate(() => {
    const memoryResult = agentResults.find(r => r.agent === 'memory');
    if (memoryResult?.result?.newInfoToSave) {
      console.log(`[기억 머니야] 저장 예정: ${memoryResult.result.newInfoToSave}`);
      // Firestore 연동 시: await updateCustomerSummary(customerData.uid, memoryResult.result);
    }
  });

  return {
    answer:      finalAnswer,
    agentsUsed:  agentsToRun,
    shadowScore,
    routing:     routing.result,
  };
}

// ════════════════════════════════════════════════════════════
//  폴백 — 단일 Claude 모드 (멀티에이전트 실패 시 자동 전환)
// ════════════════════════════════════════════════════════════
async function singleAgentChat(message, userName, financialContext, budgetInfo, conversationHistory) {
  const ragResults     = searchRAG(message, 3);
  const formulaResults = searchFormulaRAG(message, 2);
  const ragContext = [
    ragResults.map(r => `[${r.source}] ${r.topic}: ${r.content}`).join('\n\n'),
    buildFormulaContext(formulaResults),
  ].filter(Boolean).join('\n');

  const systemPrompt = createSystemPrompt(userName, financialContext, budgetInfo, ragContext);

  const response = await anthropic.messages.create({
    model: 'claude-sonnet-4-5',
    max_tokens: 400,
    system: systemPrompt,
    messages: [
      ...(conversationHistory || []).map(m => ({ role: m.role, content: m.content })),
      { role: 'user', content: message },
    ],
  });

  return response.content[0].text;
}

// ════════════════════════════════════════════════════════════
//  데이터 로드
// ════════════════════════════════════════════════════════════
loadRAGData();
loadFormulaRAG();

// ════════════════════════════════════════════════════════════
//  API 엔드포인트
// ════════════════════════════════════════════════════════════
app.get('/', (req, res) => {
  res.json({
    status: 'AI머니야 서버 실행 중!', version: '5.0',
    mode: 'multi-agent',
    agents: Object.keys(AGENT_PROMPTS).length,
    rag: {
      저서3권: ragData.books.length, AFPK: ragData.afpk.length,
      반퇴시대: ragData.bantoe.length, 명언: ragData.quotes.length,
      문제은행: ragData.questions.length, 워크북: ragData.workbook.length,
      상담사례: ragData.consultation.length, 전문강의: ragData.lecture.length,
      CFHA: ragData.cfha.length, 고객Q: ragData.custQ.length, 잔소리: ragData.nagging.length,
      공식지식베이스: formulaChunks.length,
    },
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

// ── 핵심: /api/chat — 멀티에이전트 + 폴백 ──────────────────
app.post('/api/chat', async (req, res) => {
  const { message, userName, financialContext, budgetInfo, conversationHistory } = req.body;

  if (!message) return res.json({ success: false, message: '메시지가 없습니다.' });

  // 간단한 인사는 에이전트 스킵 (비용 최적화)
  if (isSimpleMessage(message)) {
    try {
      const answer = await singleAgentChat(message, userName, financialContext, budgetInfo, conversationHistory);
      return res.json({ success: true, message: answer, meta: { mode: 'simple' } });
    } catch (e) {
      return res.json({ success: false, message: '잠시 후 다시 시도해주세요.' });
    }
  }

  // 멀티에이전트 실행
  try {
    const customerData = {
      name: userName || '고객',
      financialContext,
      budgetInfo,
      recentMessages: conversationHistory || [],
      conversation:   conversationHistory || [],
    };

    const result = await multiAgentChat(message, customerData, conversationHistory || []);

    return res.json({
      success: true,
      message: result.answer,
      meta: {
        mode:        'multi-agent',
        agentsUsed:  result.agentsUsed,
        shadowScore: result.shadowScore,
        routing:     result.routing,
      },
    });

  } catch (multiError) {
    // ── 자동 폴백: 멀티에이전트 실패 → 단일 Claude 모드 ──
    console.error('[멀티에이전트] 오류 → 단일 모드 전환:', multiError.message);
    try {
      const answer = await singleAgentChat(message, userName, financialContext, budgetInfo, conversationHistory);
      return res.json({ success: true, message: answer, meta: { mode: 'fallback', error: multiError.message } });
    } catch (fallbackError) {
      console.error('[폴백] 오류:', fallbackError.message);
      return res.json({ success: false, message: '잠시 후 다시 시도해주세요.' });
    }
  }
});

// TTS — 기존 그대로 유지
app.post('/api/tts', async (req, res) => {
  try {
    const { text, voice = 'shimmer' } = req.body;
    const response = await openai.audio.speech.create({ model: 'tts-1', voice, input: text, response_format: 'mp3' });
    const buffer = Buffer.from(await response.arrayBuffer());
    res.json({ success: true, audio: buffer.toString('base64') });
  } catch (error) { res.json({ success: false, error: 'TTS failed' }); }
});

// ════════════════════════════════════════════════════════════
//  WebSocket 음성 대화 — 기존 그대로 100% 유지
// ════════════════════════════════════════════════════════════
const PORT   = process.env.PORT || 3001;
const server = app.listen(PORT, () => console.log(`AI머니야 v5.0 멀티에이전트 서버 시작! 포트: ${PORT}`));

const wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
  console.log('[Realtime] WebSocket 연결됨');
  let openaiWs = null;
  let userName = '고객';
  let financialContext = null;
  let budgetInfo = null;

  ws.on('message', (message) => {
    try {
      const msg = JSON.parse(message);
      if (msg.type === 'start_app') {
        userName         = msg.userName || '고객';
        financialContext = msg.financialContext || null;
        budgetInfo       = msg.budgetInfo || null;
        console.log('[Realtime] 재무 정보 수신:', { name: financialContext?.name, age: financialContext?.age });

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
            if (event.type === 'response.audio.delta' && event.delta)
              ws.send(JSON.stringify({ type: 'audio', data: event.delta }));
            if (event.type === 'input_audio_buffer.speech_started')
              ws.send(JSON.stringify({ type: 'interrupt' }));
            if (event.type === 'response.audio_transcript.done') {
              console.log('머니야:', event.transcript);
              ws.send(JSON.stringify({ type: 'transcript', text: event.transcript, role: 'assistant' }));
            }
            if (event.type === 'conversation.item.input_audio_transcription.completed') {
              console.log('사용자:', event.transcript);
              ws.send(JSON.stringify({ type: 'transcript', text: event.transcript, role: 'user' }));
            }
            if (event.type === 'error') {
              console.error('OpenAI 에러:', event.error);
              ws.send(JSON.stringify({ type: 'error', error: event.error?.message }));
            }
          } catch (e) { console.error('OpenAI 메시지 파싱 에러:', e); }
        });

        openaiWs.on('error', (err) => {
          console.error('OpenAI WebSocket 에러:', err.message);
          ws.send(JSON.stringify({ type: 'error', error: err.message }));
        });
        openaiWs.on('close', () => console.log('OpenAI 연결 종료'));
      }

      if (msg.type === 'audio' && openaiWs && openaiWs.readyState === WebSocket.OPEN)
        openaiWs.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: msg.data }));

      if (msg.type === 'stop') {
        console.log('[Realtime] 종료 요청');
        if (openaiWs) openaiWs.close();
      }
    } catch (e) { console.error('메시지 처리 에러:', e); }
  });

  ws.on('close', () => {
    console.log('[Realtime] 클라이언트 연결 종료');
    if (openaiWs) openaiWs.close();
  });
});

console.log('AI머니야 v5.0 멀티에이전트 서버 초기화 완료!');
