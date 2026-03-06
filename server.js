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
- 희망: "지금 알게 되셨으니까 고칠 수 있어요!"

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
1. "부부 합산 월 소득이 세후로 얼마나 되세요?"
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
"지금 알게 되셨으니까 고칠 수 있어요! 오늘부터 시작하면 됩니다!"
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
수입 파악: "부부 합산 월 소득(세후 실수령액)이 얼마나 되세요?"
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

app.get('/', (req, res) => {
  res.json({
    status: 'AI머니야 서버 실행 중!', version: '8.0 (단일에이전트)',
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

// ════════════════════════════════════════════════════════════
//  상담탭 전용 API — 단일 에이전트 (Claude + RAG)
//  멀티에이전트 제거 → 속도 3~5배 향상, 비용 4배 절감
// ════════════════════════════════════════════════════════════
app.post('/api/consult-chat', async (req, res) => {
  try {
    const { message, userName, financialContext, conversationHistory = [] } = req.body;

    // RAG 검색
    const ragResults = searchRAG(message, 3);
    const formulaResults = searchFormulaRAG(message, 2);
    const ragContext = ragResults.map(r => `[${r.source}] ${r.topic}: ${r.content}`).join('\n\n');
    const formulaContext = buildFormulaContext(formulaResults);
    const fullRagContext = ragContext + formulaContext;

    // 시스템 프롬프트 생성
    const systemPrompt = createSystemPrompt(userName, financialContext, null, fullRagContext);

    // Claude 단일 호출
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

    res.json({
      success: true,
      message: aiText,
      panelData: null,
      meta: { ragUsed: ragResults.length, formulaUsed: formulaResults.length }
    });

  } catch (error) {
    console.error('[상담채팅] 에러:', error.status, error.message);
    res.json({ success: false, error: error.message, message: '잠시 후 다시 시도해주세요.' });
  }
});

// ════════════════════════════════════════════════════════════
//  Zoom 화상상담 API (Phase 1)
// ════════════════════════════════════════════════════════════

async function getZoomAccessToken() {
  const credentials = Buffer.from(
    `${process.env.ZOOM_CLIENT_ID}:${process.env.ZOOM_CLIENT_SECRET}`
  ).toString('base64');

  const params = new URLSearchParams();
  params.append('grant_type', 'account_credentials');
  params.append('account_id', process.env.ZOOM_ACCOUNT_ID);

  const response = await fetch('https://zoom.us/oauth/token', {
    method: 'POST',
    headers: {
      'Authorization': `Basic ${credentials}`,
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    body: params.toString()
  });

  const data = await response.json();
  console.log('[Zoom] 토큰 응답:', JSON.stringify(data));
  if (!data.access_token) throw new Error(`Zoom 토큰 발급 실패: ${JSON.stringify(data)}`);
  return data.access_token;
}

app.post('/api/zoom/create-meeting', async (req, res) => {
  try {
    const { customerName, scheduledTime, duration = 90 } = req.body;
    const token = await getZoomAccessToken();

    const meetingRes = await fetch('https://api.zoom.us/v2/users/me/meetings', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        topic: `AI머니야 재무상담 — ${customerName}님`,
        type: scheduledTime ? 2 : 1,
        start_time: scheduledTime || undefined,
        duration: duration,
        timezone: 'Asia/Seoul',
        settings: {
          join_before_host: true,
          mute_upon_entry: false,
          audio: 'both',
          auto_recording: 'cloud',
          waiting_room: false
        }
      })
    });

    const meeting = await meetingRes.json();
    console.log('[Zoom] 미팅 생성:', meeting.id);

    res.json({
      success: true,
      meetingId: meeting.id,
      joinUrl: meeting.join_url,
      startUrl: meeting.start_url,
      password: meeting.password
    });
  } catch (error) {
    console.error('[Zoom] 미팅 생성 에러:', error.message);
    res.json({ success: false, error: error.message });
  }
});

app.get('/api/zoom/meetings', async (req, res) => {
  try {
    const token = await getZoomAccessToken();
    const response = await fetch('https://api.zoom.us/v2/users/me/meetings', {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    const data = await response.json();
    res.json({ success: true, meetings: data.meetings || [] });
  } catch (error) {
    console.error('[Zoom] 미팅 목록 에러:', error.message);
    res.json({ success: false, error: error.message });
  }
});

// ElevenLabs TTS
async function elevenLabsTTS(text) {
  const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
  const VOICE_ID = process.env.ELEVENLABS_VOICE_ID;
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}/stream`,
    {
      method: 'POST',
      headers: { 'xi-api-key': ELEVENLABS_API_KEY, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        model_id: 'eleven_multilingual_v2',
        voice_settings: { stability: 0.5, similarity_boost: 0.75 },
        output_format: 'mp3_44100_128',
      }),
    }
  );
  if (!response.ok) throw new Error(`ElevenLabs 에러: ${response.status}`);
  return Buffer.from(await response.arrayBuffer());
}

app.post('/api/consult-tts', async (req, res) => {
  try {
    const { text } = req.body;
    const buffer = await elevenLabsTTS(text);
    res.json({ success: true, audio: buffer.toString('base64') });
  } catch (error) {
    console.error('[상담TTS] ElevenLabs 에러:', error);
    try {
      const { text } = req.body;
      const fallback = await openai.audio.speech.create({ model: 'tts-1', voice: 'onyx', input: text, response_format: 'mp3' });
      const buffer = Buffer.from(await fallback.arrayBuffer());
      res.json({ success: true, audio: buffer.toString('base64'), fallback: true });
    } catch {
      res.json({ success: false, error: 'TTS 실패' });
    }
  }
});

const PORT = process.env.PORT || 3001;
const server = app.listen(PORT, () => console.log(`AI머니야 서버 v8.0 시작! 포트: ${PORT}`));

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

  ws.on('message', async (message) => {
    try {
      const msg = JSON.parse(message);

      // 상담탭: OpenAI Realtime (음성)
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
              console.log('[상담WS] 머니야:', event.transcript?.slice(0, 50));
              ws.send(JSON.stringify({ type: 'transcript', text: event.transcript, role: 'assistant' }));
            }
            if (event.type === 'conversation.item.input_audio_transcription.completed') {
              console.log('[상담WS] 사용자:', event.transcript);
              ws.send(JSON.stringify({ type: 'transcript', text: event.transcript, role: 'user' }));
            }
            if (event.type === 'error') {
              console.error('[상담WS] OpenAI 에러:', event.error);
              ws.send(JSON.stringify({ type: 'error', error: event.error?.message }));
            }
          } catch (e) { console.error('[상담WS] 메시지 파싱 에러:', e); }
        });

        openaiWs.on('error', (err) => {
          console.error('[상담WS] OpenAI 에러:', err.message);
          ws.send(JSON.stringify({ type: 'error', error: err.message }));
        });
        openaiWs.on('close', () => console.log('[상담WS] OpenAI 연결 종료'));
        return;
      }

      // AI지출탭: OpenAI Realtime
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
        openaiWs.on('error', (err) => { console.error('OpenAI WebSocket 에러:', err.message); ws.send(JSON.stringify({ type: 'error', error: err.message })); });
        openaiWs.on('close', () => console.log('OpenAI 연결 종료'));
      }

      if (msg.type === 'audio' && openaiWs && openaiWs.readyState === WebSocket.OPEN) {
        openaiWs.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: msg.data }));
      }
      if (msg.type === 'stop') {
        console.log('[WS] 종료 요청');
        if (openaiWs) openaiWs.close();
      }
    } catch (e) { console.error('메시지 처리 에러:', e); }
  });

  ws.on('close', () => { console.log('[WS] 클라이언트 연결 종료'); if (openaiWs) openaiWs.close(); });
});

console.log('AI머니야 서버 v8.0 초기화 완료!');
