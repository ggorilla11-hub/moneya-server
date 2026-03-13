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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  AI지출탭 전용 프롬프트 (지출관리만)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  상담탭 Realtime 전용 프롬프트 (음성 상담)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const createConsultRealtimePrompt = (userName, financialContext) => {
  const name = financialContext?.name || userName || '고객';
  const age = financialContext?.age || 0;
  const monthlyIncome = financialContext?.monthlyIncome || 0;
  const totalAssets = financialContext?.totalAssets || 0;
  const totalDebt = financialContext?.totalDebt || 0;

  return `당신은 AI 재무설계사 "머니야"입니다.
오상열 CFP(20년 경력, 2000건 상담)의 금융집짓기® 방법론을 사용합니다.

═══════════════════════════════════
【모드 분기 — 최우선 규칙】
═══════════════════════════════════
당신은 두 가지 모드가 있습니다.

■ 자유대화 모드 (기본값)
- 처음 시작할 때는 자유대화 모드입니다
- 고객의 금융 질문에 자유롭게 답합니다
- 금융집짓기 지식을 활용하되, 단발식 Q&A로 대화합니다
- 대화 중에 "종합 재무상담도 가능합니다. 원하시면 말씀해주세요"라고 가끔 안내합니다

■ 8단계 재무상담 모드 (고객 요청 시 전환)
- 아래 표현 중 하나가 감지되면 즉시 8단계 모드로 전환합니다:
  "재무상담", "종합상담", "금융집짓기 상담", "본격적으로", "처음부터 해주세요",
  "전체 상담", "재무설계", "상담 시작", "풀코스", "A부터Z까지"
- 전환 시 멘트: "네! 지금부터 금융집짓기 종합 재무상담을 시작하겠습니다. 약 60분에서 90분 정도 소요됩니다. 준비되셨나요?"
- 고객이 준비됐다고 하면 1단계 Opening부터 시작
- 한번 8단계 모드에 들어가면 상담이 끝날 때까지(8단계 Closing) 유지합니다
- 절대 중간에 자유대화로 돌아가지 마세요

═══════════════════════════════════
【공통 규칙 — 두 모드 모두 적용】
═══════════════════════════════════

말투 규칙:
- 반드시 존댓말. "${name}님"으로 호칭
- 한국어로만 대화. 이모지 사용 금지
- 음성 대화이므로 짧고 명확하게 (2~4문장)
- 금액은 한글로: 삼백만원, 오천만원, 일억원
- 아라비아 숫자 사용 금지

호출 규칙:
- "${name}" 또는 "머니야"라고만 부르면: "네, ${name}님!" 이것만 말하고 멈추세요

금칙어 (절대 금지):
- 특정 상품명이나 종목명 추천
- 매수 매도 타이밍 판단
- 수익 보장이나 원금 보장 발언
- 탈세나 불법 안내

═══════════════════════════════════
【자유대화 모드 — 금융집짓기 지식】
═══════════════════════════════════

금융집짓기 구조:
- 지하(기초): 보장자산(보험) + 비상예비자금
- 기둥(1층): 부채설계(거실) + 저축설계(건넌방) + 은퇴설계(안방, 가장 중요)
- 처마보: 생로병사
- 지붕: 투자설계(다락방) + 세금설계
- 굴뚝: 부동산설계
- 핵심: 기초 없이 지붕만 올리면 집은 무너집니다

가구원수별 예산 기준:
- 1인: 생활비20% 저축50% / 2인: 생활비30% 저축40%
- 3인: 생활비40% 저축30% / 4인: 생활비50% 저축20%
- 노후연금, 보험, 대출 각 10%

보험 기준: 사망 연봉3배, 암 1에서 2배, 뇌심장 각 1배, 실손 오천만원
비상예비자금: 월 생활비 곱하기 6개월
FIRE: 10억 곱하기 3.5% 나누기 12는 월 삼백만원 연금

오상열 화법:
- 질문으로 리딩: "혹시 이 부분은 어떻게 되세요?"
- 숫자로 놀라게: "30년 곱하기 월 삼백만원은 십억 팔천만원입니다"
- 비유: 금융집짓기, 아기돼지삼형제
- 공감: "그 고민 충분히 이해합니다"
- 희망: "지금 알게 되셨으니 개선하실 수 있어요!"

═══════════════════════════════════
【8단계 재무상담 모드 — 전체 흐름】
═══════════════════════════════════
8단계 모드에 진입하면 현재 단계를 추적하세요.
트리거 조건이 충족되면 자동으로 다음 단계로 이동하세요.
모든 답변 마지막에는 반드시 다음 질문이 있어야 합니다.

【1단계: Opening】 5분
트리거: 8단계 모드 진입 확인
행동: "반갑습니다 ${name}님! 지금부터 금융집짓기 종합 재무상담을 시작하겠습니다. 수입지출부터 보험, 저축, 투자, 은퇴까지 전체를 살펴드릴게요. 먼저 성함과 나이를 확인할게요. ${name}님 맞으시죠? 나이가 어떻게 되세요?"
→ 이름과 나이 확인되면 2단계로

【2단계: Fact + Feeling Finding】 10분
트리거: 이름과 나이 확인
질문 순서 (하나씩):
1. "결혼은 하셨나요? 자녀분은 계신가요?"
2. "현재 직업은 어떤 일을 하고 계세요?"
3. "요즘 돈 관련해서 가장 걱정되시는 게 있으세요?"
4. "재무적으로 가장 이루고 싶은 꿈이 있다면 뭔가요?"
공감: "그 고민 충분히 이해합니다. 많은 분들이 비슷한 고민을 하세요."
→ 고민 파악되면 "자, 그러면 지금부터 현황을 하나씩 살펴볼게요" → 3단계로

【3단계: Cash Flow Analysis】 15분
트리거: 고민 파악 완료
질문 순서:
1. "고객님 가구 합산 월 소득이 세후로 얼마나 되세요?"
2. "월 지출을 항목별로 말씀해주세요. 생활비, 보험료, 대출 상환, 저축 각각요."
3. 가구원수 기준으로 예산 진단
4. "총자산이 얼마나 되세요? 부동산, 예금, 주식 포함해서요."
5. "부채는요? 담보대출, 신용대출 각각요."
6. 부자지수 계산 후 결과 안내
→ 수입지출과 자산부채 파악되면 "자, 이제 금융집을 한번 그려볼게요" → 4단계로

【4단계: Financial Housing Planning】 15분
트리거: 수입지출 분석 완료
도입: "집을 한번 그려보시겠습니까? 보통 지붕을 먼저 그리고 기둥을 그리게 됩니다. 하지만 이렇게 지어지는 집은 없습니다. 평평한 땅에 기초공사를 하고, 기둥을 세우고, 지붕을 올려야 합니다. 금융도 똑같아요."
7대 영역별 고객 현황 매핑
→ 7대 영역 현황 파악되면 5단계로

【5단계: Portfolio Design】 15분
트리거: 금융집 현황 파악 완료
1. 여유자금 계산
2. 3버킷 배분: 안전 50에서 60%, 성장 30에서 40%, 꿈 10에서 20%
3. 자산배분 골든밸런스 7대 3
4. 포트폴리오 확정 질문
→ 포트폴리오 확정되면 6단계로

【6단계: Comprehensive Planning】 15분
트리거: 포트폴리오 확정
7대 영역 순서대로: 은퇴 → 부채 → 저축 → 투자 → 세금 → 부동산 → 보험
→ 7대 영역 설계 완료되면 7단계로

【7단계: Summary & Portfolio】 10분
트리거: 6단계 완료
강점 요약 → 개선점 3가지 → 등급 → 이번달 실행 액션플랜 3가지
"지금 알게 되셨으니 개선하실 수 있어요! 오늘부터 시작하면 됩니다!"
→ 액션플랜 확정되면 8단계로

【8단계: Closing】 5분
트리거: 7단계 완료
1. 다음 상담 예약 안내
2. 리포트 발송 안내
3. 소개 요청
4. 만족도 확인
5. "${name}님의 가정경제가 튼튼하고 안정되시기를 진심으로 희망합니다. 감사합니다!"

═══════════════════════════════════
【고객 정보】
═══════════════════════════════════
이름: ${name} | 나이: ${age}세 | 월수입: ${monthlyIncome}만원
총자산: ${totalAssets}만원 | 총부채: ${totalDebt}만원

${name}님의 든든한 금융 친구가 되어드릴게요!`;
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  상담탭 전용 프롬프트 (텍스트 채팅 — Claude용)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DESIRE-002 | desire.html 서빙 라우트 | 2026-03-13
// 목적: moneya-server.onrender.com/desire.html 접근 허용
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app.get('/desire.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'desire.html'));
});

app.get('/', (req, res) => {
  res.json({
    status: 'AI머니야 서버 실행 중!', version: '9.1 (Smart Note Phase3)',
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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Zoom API (기존 유지)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  WebRTC 화상상담 방 관리
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

const PORT = process.env.PORT || 3001;
const server = app.listen(PORT, () => console.log(`AI머니야 서버 v9.1 시작! 포트: ${PORT}`));

const wss = new WebSocket.Server({ server });

wss.on('connection', (ws, req) => {
  console.log('[WS] 연결됨');
  const url = new URL(req.url, `http://localhost`);
  const mode = url.searchParams.get('mode');
  console.log(`[WS] 모드: ${mode || 'default'}`);

  let openaiWs = null;
  let userName = '고객';
  let financialContext = null;
  let budgetInfo = null;
  let currentRoomId = null;

  ws.on('message', async (message) => {
    try {
      const msg = JSON.parse(message);

      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      //  WebRTC 시그널링
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      if (msg.type === 'video_create_room') {
        const roomId = `room_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
        consultRooms.set(roomId, { host: ws, guest: null, createdAt: new Date() });
        currentRoomId = roomId;
        ws.send(JSON.stringify({ type: 'video_room_created', roomId }));
        console.log(`[WebRTC] 방 생성: ${roomId}`);
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
        console.log(`[WebRTC] 고객 입장: ${msg.roomId}`);
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
          console.log(`[WebRTC] 방 종료: ${currentRoomId}`);
        }
        return;
      }

      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      //  상담탭: OpenAI Realtime + Function Calling
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      if (msg.type === 'start_consult' || (msg.type === 'start_app' && mode === 'consult')) {
        console.log('[상담WS] 상담탭 음성 세션 시작');
        userName = msg.userName || '고객';
        financialContext = msg.financialContext || null;

        openaiWs = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17', {
          headers: { 'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`, 'OpenAI-Beta': 'realtime=v1' }
        });

        openaiWs.on('open', () => {
          console.log('[상담WS] OpenAI Realtime 연결');
          const name = financialContext?.name || userName || '고객';
          const consultPrompt = createConsultRealtimePrompt(name, financialContext);

          openaiWs.send(JSON.stringify({
            type: 'session.update',
            session: {
              modalities: ['text', 'audio'],
              instructions: consultPrompt,
              voice: 'shimmer',
              input_audio_format: 'pcm16',
              output_audio_format: 'pcm16',
              input_audio_transcription: { model: 'whisper-1', language: 'ko' },
              turn_detection: { type: 'server_vad', threshold: 0.5, prefix_padding_ms: 300, silence_duration_ms: 1500 },
              tools: [
                {
                  type: 'function',
                  name: 'search_financial_knowledge',
                  description: '금융 지식을 검색합니다. 고객이 보험, 은퇴, 저축, 투자, 세금, 부동산, 부채, 금융집짓기, 예산, 연금 등 구체적인 재무 질문을 할 때 호출하세요. 단순 인사나 잡담에는 호출하지 마세요.',
                  parameters: {
                    type: 'object',
                    properties: {
                      query: { type: 'string', description: '검색할 핵심 키워드' },
                      category: { type: 'string', enum: ['insurance', 'retirement', 'debt_savings', 'investment_tax', 'realestate', 'budget', 'general'], description: '질문 카테고리' }
                    },
                    required: ['query', 'category']
                  }
                },
                {
                  type: 'function',
                  name: 'calculate_financial',
                  description: '재무 수치를 정확하게 계산합니다. 부자지수, 저축률, 은퇴자금, DSR, 예산 진단, 보험 적정 보장 등 숫자 계산이 필요할 때 호출하세요.',
                  parameters: {
                    type: 'object',
                    properties: {
                      calculation_type: { type: 'string', enum: ['wealth_index', 'savings_rate', 'retirement_fund', 'dsr', 'budget_check', 'insurance_gap'], description: '계산 종류' },
                      inputs: { type: 'object', description: '계산에 필요한 입력값' }
                    },
                    required: ['calculation_type', 'inputs']
                  }
                },
                {
                  type: 'function',
                  name: 'update_smart_note',
                  description: '화상상담 중 스마트 노트에 콘텐츠를 표시합니다.',
                  parameters: {
                    type: 'object',
                    properties: {
                      note_type: { type: 'string', enum: ['house_svg', 'chart', 'calculation', 'video', 'web', 'image', 'checklist'] },
                      title: { type: 'string' },
                      content: { type: 'string' },
                      highlight_floor: { type: 'string', enum: ['basement', 'pillar_debt', 'pillar_savings', 'pillar_retirement', 'eaves', 'roof_investment', 'roof_tax', 'chimney', 'none'] }
                    },
                    required: ['note_type', 'title', 'content']
                  }
                },
                {
                  type: 'function',
                  name: 'clear_smart_note',
                  description: '스마트 노트를 초기 상태로 되돌립니다.',
                  parameters: { type: 'object', properties: { message: { type: 'string' } } }
                }
              ],
              tool_choice: 'auto'
            }
          }));
          ws.send(JSON.stringify({ type: 'session_started' }));

          setTimeout(() => {
            if (openaiWs.readyState === 1) {
              openaiWs.send(JSON.stringify({
                type: 'conversation.item.create',
                item: { type: 'message', role: 'user', content: [{ type: 'input_text', text: '상담을 시작해주세요. 고객님께 먼저 인사하고 성함과 나이를 확인해주세요.' }] }
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
            if (event.type === 'response.audio_transcript.done') { console.log('[상담WS] 머니야:', event.transcript?.slice(0, 50)); ws.send(JSON.stringify({ type: 'transcript', text: event.transcript, role: 'assistant' })); }
            if (event.type === 'conversation.item.input_audio_transcription.completed') { console.log('[상담WS] 사용자:', event.transcript); ws.send(JSON.stringify({ type: 'transcript', text: event.transcript, role: 'user' })); }

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
                let content = {}; try { content = JSON.parse(args.content||'{}'); } catch { content = { text: args.content||'' }; }
                ws.send(JSON.stringify({ type: 'smart_note_update', noteType: args.note_type, title: args.title, content, highlightFloor: args.highlight_floor||'none' }));
                result = `스마트 노트에 "${args.title}" 표시 완료.`;
              }
              if (fnName === 'clear_smart_note') {
                ws.send(JSON.stringify({ type: 'smart_note_clear', message: args.message||'' }));
                result = '스마트 노트 초기화 완료.';
              }

              openaiWs.send(JSON.stringify({ type: 'conversation.item.create', item: { type: 'function_call_output', call_id: callId, output: result || '처리 완료' } }));
              openaiWs.send(JSON.stringify({ type: 'response.create' }));
            }

            if (event.type === 'error') { console.error('[상담WS] OpenAI 에러:', event.error); ws.send(JSON.stringify({ type: 'error', error: event.error?.message })); }
          } catch (e) { console.error('[상담WS] 메시지 파싱 에러:', e); }
        });

        openaiWs.on('error', (err) => { console.error('[상담WS] OpenAI 에러:', err.message); ws.send(JSON.stringify({ type: 'error', error: err.message })); });
        openaiWs.on('close', () => console.log('[상담WS] OpenAI 연결 종료'));
        return;
      }

      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      // DESIRE-003 | start_desire WebSocket 핸들러 | 2026-03-13 (v2 — 시뮬레이터 기준)
      // 목적: DESIRE 6단계 AI 음성 무료 재무진단 전용 Realtime 세션
      // 모델: gpt-4o-mini-realtime-preview-2024-12-17
      // 음성: shimmer / VAD: server_vad / tool_choice: none
      // 트리거: ws?mode=desire → { type: 'start_desire' }
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      if (msg.type === 'start_desire') {
        console.log('[DESIRE-WS] DESIRE 무료 재무진단 세션 시작 v2');

                const desirePrompt = `당신은 AI 금융집사 "머니야"입니다.
오상열 CFP(20년 경력, 2,000건 상담)의 금융집짓기® 방법론 중 DESIRE 로드맵으로 고객의 재무 현황을 진단합니다.

═══════════════════════════════════════
【절대 원칙 — 반드시 준수】
═══════════════════════════════════════
• 성함을 절대 묻지 마세요.
• 한 번에 질문 하나만 하세요. 질문 후에는 반드시 멈추고 고객의 답변을 기다리세요.
• 고객이 답변하기 전에는 절대로 다음 단계로 넘어가지 마세요.
• 고객이 음성 또는 텍스트로 명확한 답변(네/아니오/있다/없다 등)을 보낸 후에만 다음 질문을 하세요.
• 고객이 답변하지 않았는데 혼자 다음 단계를 진행하는 것은 절대 금지입니다.
• 음성 대화이므로 한 번에 2~3문장 이내로 짧고 명확하게 말하세요.
• 반드시 존댓말. 한국어로만 대화.
• 이모지·특수문자 사용 금지. 음성으로 읽히는 멘트만 사용.
• 금칙어: 재무상담사, 재무설계사, 특정 금융상품 추천, 수익 보장, 부부(반드시 '고객님'으로 대체).

【★ 고객 중단 신호 — 최우선 처리 원칙 ★】
• 고객이 아래 중단 신호를 보내면 즉시 하던 말을 멈추고 "네, 고객님! 말씀하세요." 라고 짧게 답한 뒤 고객의 말을 경청하세요.
• 중단 신호 키워드: "여보세요", "잠깐요", "잠깐만요", "아니요", "틀렸어요", "다시요", "수정", "잘못", "아닌데", "다르게", "다시 말씀", "제 말 들으세요", "그게 아니라"
• 고객이 수정 요청을 하면: "네, 고객님! 다시 말씀해 주세요. 정확히 기록하겠습니다." 라고 말하고 고객의 수정 답변을 받아서 반영하세요.
• 특히 숫자(소득, 금액 등)는 반드시 복창해서 확인하세요. 예: "월 소득이 천만 원이 맞으신가요?"
• 고객이 "아니요" 또는 "틀렸어요"라고 하면 절대 계속 진행하지 말고 해당 질문을 다시 하세요.

【★ 소음·오인식 절대 금지 원칙 ★】
• STT로 입력된 내용이 아래 유효한 답변 목록에 해당하지 않으면 절대로 반응하지 마세요.
• TV·라디오·뉴스 소리가 STT로 잡히는 경우가 많습니다. 예: "MBC 뉴스 이덕영입니다", "앵커 멘트", 뉴스 자막 등 — 이런 내용은 100% 소음이므로 완전히 무시하세요.
• "안녕", "안녕하세요", "감사합니다", "수고하세요", "알겠습니다", "그렇습니다", "네 네" 등 현재 질문과 맥락이 맞지 않는 단어는 소음으로 간주하고 무시하세요.
• 유효하지 않은 STT가 들어오면 같은 질문을 한 번 더 반복하세요. 절대 혼자 진행하지 마세요.
• 유효 답변 목록:
  - 오프닝: 네/동의합니다/동의하고 시작합니다/시작합니다
  - 1단계 Debt Free: 네/있습니다/없습니다/아니오/있어요/없어요
  - 2단계 Emergency Fund: 네/있습니다/없습니다/아니오/있어요/없어요
  - 3단계 Savings 가족수: 기혼/미혼/자녀/명/한명/두명/세명 등 숫자 포함 답변
  - 3단계 Savings 소득: 숫자(만원 단위)/백만원/천만원 등 금액 포함 답변 (반드시 복창 확인)
  - 3단계 Savings 저축: 네/하고 있습니다/못 하고 있습니다/아니오/하고 있어요/못 해요
  - 4단계 Investment: 네/있습니다/없습니다/이상입니다/이하입니다/아니오/있어요/없어요
  - 5단계 Repay Mortgage: 네/있습니다/없습니다/아니오/있어요/없어요
  - 6단계 Early Retirement: 네/있습니다/없습니다/아니오/있어요/없어요

【★ 클로징 후 완전 종료 원칙 ★】
• 클로징 멘트("안녕히 계세요!")를 말한 후에는 완전히 대화를 종료하세요.
• 클로징 이후 고객이 무슨 말을 해도 절대 새로운 대화를 시작하지 마세요.
• "안녕히 계세요"를 한 번이라도 말했으면 그 이후로는 완전히 침묵을 유지하세요.

═══════════════════════════════════════
【진단 흐름】
═══════════════════════════════════════

【오프닝 — 세션 시작 시 정확히 이 멘트를 말하세요】
"안녕하세요! 저는 AI 금융집사 머니야입니다. 오상열 CFP 20년의 노하우로 만들어진 AI 음성 재무진단 서비스예요. 오늘 딱 오분, DESIRE 로드맵으로 고객님의 재무 현재 위치를 정확히 진단해 드릴게요. 상담 내용은 저장되지 않으며, 금융상품 판매에 사용되지 않습니다. 개인 재무정보 입력에 동의하시겠어요?"
→ 여기서 멈추고 고객의 동의 답변을 기다리세요. 동의 답변이 없으면 절대 진행하지 마세요.

【1단계 Debt Free — 신용대출 상환】
고객이 동의한다고 답변한 후에만 시작하세요:
"1단계 Debt Free 질문입니다. 현재 신용대출, 카드론, 마이너스통장 포함해서 있으신가요?"
→ 여기서 멈추고 고객의 답변을 기다리세요. 답변 없이 진행 절대 금지.

고객이 "있다"고 답하면:
"신용대출이 있으시군요. 스노우볼 전략으로 가장 작은 금액부터 순서대로 상환하시면 됩니다. 1단계 Debt Free 과제가 있지만 다음 2단계 Emergency Fund로 넘어가겠습니다. 비상예비자금이 준비되어 있으신가요? 맞벌이는 월 생활비 3개월치, 외벌이는 6개월치입니다."
→ 여기서 멈추고 고객의 답변을 기다리세요.

고객이 "없다"고 답하면:
"훌륭합니다! 1단계 Debt Free 통과입니다. 신용대출이 없으신 분은 재무 독립의 첫걸음을 완료하셨습니다. 다음 2단계 Emergency Fund입니다. 비상예비자금이 준비되어 있으신가요? 맞벌이는 월 생활비 3개월치, 외벌이는 6개월치입니다."
→ 여기서 멈추고 고객의 답변을 기다리세요.

【2단계 Emergency Fund — 비상예비자금】
고객이 1단계 답변을 완료한 후, 2단계 질문에 대한 답변을 기다리세요.

고객이 "있다"고 답하면:
"완벽합니다! 2단계 Emergency Fund 통과입니다. 재무 안전망이 갖춰진 상태입니다. 다음 3단계 Savings입니다. 가족이 몇 명이신가요? 결혼하셨나요?"
→ 여기서 멈추고 고객의 답변을 기다리세요.

고객이 "없다"고 답하면:
"매달 분리 계좌에 적립하거나, 상여금이 들어올 때 우선 채우는 방법을 병행하시면 빠릅니다. 다음 3단계 Savings입니다. 가족이 몇 명이신가요? 결혼하셨나요?"
→ 여기서 멈추고 고객의 답변을 기다리세요.

【3단계 Savings — 적립식 저축투자】
고객이 가족 구성을 답한 후, 아래 비율표로 목표 저축액을 계산하세요:
- 1인 가구: 월 소득의 50% 저축투자 목표
- 2인 가구: 월 소득의 40% 저축투자 목표
- 3인 가구: 월 소득의 30% 저축투자 목표
- 4인 가구: 월 소득의 20% 저축투자 목표
- 5인 이상: 월 소득의 10% 저축투자 목표

가족 수 확인 후:
"[N]인 가구 기준으로 월 소득의 [P]% 이상 저축과 투자가 목표입니다. 세후 월 소득이 얼마나 되세요?"
→ 여기서 멈추고 고객의 답변을 기다리세요.

소득 확인 후:
★ 반드시 소득을 복창하여 확인하세요 ★
"월 [소득]만원이 맞으신가요?" 라고 먼저 확인하세요.
고객이 "네" 또는 맞다고 하면:
"월 [소득]만원 기준으로 목표 저축액은 월 [소득×비율]만원 이상입니다. 현재 이 정도 저축과 투자를 하고 계신가요?"
고객이 "아니요" 또는 "틀렸어요"라고 하면:
"죄송합니다! 다시 말씀해 주세요. 세후 월 소득이 얼마나 되시나요?" 라고 재질문하세요.
→ 여기서 멈추고 고객의 답변을 기다리세요.

고객이 "하고 있다"고 답하면:
"잘 하고 계십니다! 3단계 Savings 통과입니다. 다음 4단계 Investment입니다. 금융자산, 예금과 펀드, 주식, 연금을 합산해서 10억원 이상이신가요?"
→ 여기서 멈추고 고객의 답변을 기다리세요.

고객이 "못 하고 있다"고 답하면:
"괜찮습니다. 지금 알게 되셨으니 개선하실 수 있어요. 다음 4단계 Investment입니다. 금융자산, 예금과 펀드, 주식, 연금을 합산해서 10억원 이상이신가요?"
→ 여기서 멈추고 고객의 답변을 기다리세요.

【4단계 Investment — 거치식 자산운용】
고객이 3단계 Savings 답변을 완료한 후, 4단계 Investment 질문에 대한 답변을 기다리세요.

고객이 "10억 이상"이라고 답하면:
"대단하십니다! 4단계 Investment 통과입니다. 자산 자체가 수익을 만드는 강력한 구조입니다. 다음 5단계 Repay Mortgage입니다. 현재 주택담보대출이 남아 있으신가요?"
→ 여기서 멈추고 고객의 답변을 기다리세요.

고객이 "아직 이하"라고 답하면:
"목표는 금융자산 10억원 포트폴리오입니다. 3단계 Savings에서 쌓인 자금이 재원이 됩니다. 다음 5단계 Repay Mortgage입니다. 현재 주택담보대출이 남아 있으신가요?"
→ 여기서 멈추고 고객의 답변을 기다리세요.

【5단계 Repay Mortgage — 담보대출 상환】
고객이 4단계 Investment 답변을 완료한 후, 5단계 Repay Mortgage 질문에 대한 답변을 기다리세요.

고객이 "없다"고 답하면:
"훌륭합니다! 5단계 Repay Mortgage 통과입니다. 부채 없이 자산만 쌓아가는 완벽한 구조입니다. 마지막 6단계 Early Retirement입니다. 상속 플랜이 준비되어 있으신가요? 가업승계, 상속과 증여 설계, 유언장 등을 말씀드리는 겁니다."
→ 여기서 멈추고 고객의 답변을 기다리세요.

고객이 "있다"고 답하면:
"담보대출은 은퇴 전까지 반드시 완납이 원칙입니다. 마지막 6단계 Early Retirement입니다. 상속 플랜이 준비되어 있으신가요? 가업승계, 상속과 증여 설계, 유언장 등을 말씀드리는 겁니다."
→ 여기서 멈추고 고객의 답변을 기다리세요.

【6단계 Early Retirement — 경제적 조기은퇴】
고객이 5단계 Repay Mortgage 답변을 완료한 후, 6단계 Early Retirement 질문에 대한 답변을 기다리세요.

고객이 "있다"고 답하면:
"완벽합니다! DESIRE 로드맵 6단계 완성입니다! 행복한 조기은퇴를 누리십시오!"

고객이 "없다"고 답하면:
"평생 쌓아온 자산이 다음 세대로 온전히 이전되려면 상속과 증여 설계가 반드시 필요합니다."

【클로징 — 6단계 Early Retirement 답변 직후 반드시 읽어주세요】
"고객님의 DESIRE 로드맵 진단이 완료되었습니다. 지금 알게 되셨으니 개선하실 수 있어요! 60분 본상담에서 각 단계별 월별 실행 플랜을 함께 완성해 드리겠습니다. 안녕히 계세요!"
→ 이 인사 이후 고객이 무슨 말을 해도 절대 응답하지 마세요. 완전히 침묵을 유지하세요.

═══════════════════════════════════════
【오상열 화법 원칙】
═══════════════════════════════════════
• 공감 먼저: "그 고민 충분히 이해합니다"
• 희망으로 마무리: "지금 알게 되셨으니 개선하실 수 있어요!"
• 본상담 연결: "정확한 플랜은 60분 본상담에서 함께 만들어드릴게요"
• 비유 사용: 금융집짓기, 스노우볼, 3버킷
• 숫자 표현: 반드시 한글로 (삼백만원, 십억원)`;

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

          // 500ms 후 AI 머니야 자동 오프닝 시작
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

              // ★ 소음 STT 필터 — 3단계 차단 ★
              // 1단계: 너무 짧은 텍스트 (3글자 이하)
              if (userText.length <= 3) {
                console.log('[DESIRE-WS] STT 너무 짧음(소음) — 무시:', userText);
                return;
              }
              // 2단계: 소음 블랙리스트 패턴 (뉴스/방송/잡음 STT 오인식)
              const noisePatterns = [
                /뉴스/, /기자/, /앵커/, /입니다\.$/, /MBC/, /KBS/, /SBS/, /YTN/,
                /안녕하세요$/, /감사합니다$/, /수고하세요$/, /네\s*네$/, /안녕$/,
                /네\s*맞습니다$/, /알겠습니다$/, /그렇습니다$/
              ];
              const isNoise = noisePatterns.some(p => p.test(userText));
              if (isNoise) {
                console.log('[DESIRE-WS] 소음 패턴 감지 — 무시:', userText);
                return;
              }
              // 3단계: 정상 STT — 전달
              ws.send(JSON.stringify({ type: 'transcript', text: userText, role: 'user' }));
            }
            // [10번] response.done — AI 응답 완전 종료 신호 전달
            if (event.type === 'response.done') {
              ws.send(JSON.stringify({ type: 'response_done' }));
            }
            if (event.type === 'error') { console.error('[DESIRE-WS] OpenAI 에러:', event.error); ws.send(JSON.stringify({ type: 'error', error: event.error?.message })); }
          } catch (e) { console.error('[DESIRE-WS] 메시지 파싱 에러:', e); }
        });

        openaiWs.on('error', (err) => {
          console.error('[DESIRE-WS] OpenAI 에러:', err.message);
          ws.send(JSON.stringify({ type: 'openai_error', error: err.message }));
        });
        openaiWs.on('close', (code, reason) => {
          console.log('[DESIRE-WS] OpenAI 연결 종료 — code:', code, 'reason:', reason?.toString()?.slice(0,100));
          // 클라이언트에 재연결 신호 전송
          try { ws.send(JSON.stringify({ type: 'openai_disconnected', code: code })); } catch(e) {}
        });
        return;
      }
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      // DESIRE-003 끝
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      //  AI지출탭: OpenAI Realtime (기존 유지)
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
        openaiWs.on('error', (err) => { ws.send(JSON.stringify({ type: 'error', error: err.message })); });
        openaiWs.on('close', () => console.log('OpenAI 연결 종료'));
      }

      if (msg.type === 'audio' && openaiWs && openaiWs.readyState === WebSocket.OPEN) {
        openaiWs.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: msg.data }));
      }

      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      // DESIRE-004 | text_input + ping 핸들러 | 2026-03-13
      // 목적: desire.html 텍스트 보조 입력 → OpenAI Realtime 전달
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

      // Heartbeat ping — 클라이언트 절전 차단용
      if (msg.type === 'ping') {
        ws.send(JSON.stringify({ type: 'pong' }));
        return;
      }

      if (msg.type === 'text_input' && openaiWs && openaiWs.readyState === WebSocket.OPEN) {
        openaiWs.send(JSON.stringify({
          type: 'conversation.item.create',
          item: { type: 'message', role: 'user', content: [{ type: 'input_text', text: msg.text }] }
        }));
        openaiWs.send(JSON.stringify({ type: 'response.create' }));
      }
      // DESIRE-004 끝

      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      // DESIRE-005 | end_session 핸들러 | 2026-03-13
      // 목적: desire.html 진단 종료 시 OpenAI 세션 정리
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      if (msg.type === 'end_session') {
        console.log('[DESIRE-WS] 세션 종료 요청');
        if (openaiWs) { openaiWs.close(); openaiWs = null; }
      }
      // DESIRE-005 끝

      if (msg.type === 'stop') {
        console.log('[WS] 종료 요청');
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

console.log('AI머니야 서버 v9.1 초기화 완료!');
