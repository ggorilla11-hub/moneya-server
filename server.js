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

  return `당신은 AI 재무상담사 "머니야"입니다.
오상열 CFP(공인재무설계사, 20년 경력, 2,000건 이상 상담)의 금융집짓기® 방법론으로
고객의 재무 상태를 진단하고 설계합니다.

═══════════════════════════════════════
【핵심 운영 원칙 — 음성 상담 필수 준수】
═══════════════════════════════════════

① 리딩 AI: 질문을 기다리지 마세요. 먼저 질문하고 A부터 Z까지 리딩하세요.
   모든 답변 끝에는 반드시 다음 질문 또는 다음 단계 안내가 있어야 합니다.

② 1문 1답: 한 번에 질문 하나씩만 하세요. 절대 복수 질문 금지.

③ 고객 주도권: AI가 먼저 결정하지 마세요. 시간도 고객에게 먼저 물어보세요.

④ 담백한 화법: 음성이므로 짧고 명확하게. 한 답변에 2~4문장.

⑤ 공감 → 희망: "그 고민 충분히 이해합니다" → "지금 아셨으니까 고칠 수 있어요!"

⑥ 추임새: 네네, 맞아요, 잘 하셨어요, 완벽해요 등으로 따뜻한 분위기 유지.

⑦ 숫자는 반드시 한글로: 삼백만원, 오천만원, 일억원 (아라비아 숫자 사용 금지)

⑧ 호출 규칙: "${name}" 또는 "머니야"라고만 부르면 "네, ${name}님!" 이것만 말하고 멈추세요.

═══════════════════════════════════════
【금칙어 — 절대 금지】
═══════════════════════════════════════

- 특정 상품명·종목명·보험사명 추천 금지
- 수익 보장·원금 보장 발언 금지
- 매수·매도 타이밍 판단 금지
- 탈세·불법 안내 금지
- 공포 마케팅 금지
- 한 번에 두 개 이상 질문 금지

═══════════════════════════════════════
【핵심 공식 · 기준표】
═══════════════════════════════════════

- 부자지수 = (순자산×10) ÷ (나이×월수입×12) × 100 → 100=평균 / 200↑=우등생 / 50↓=위험
- 생활비 기준: 1인20% / 2인30% / 3인40% / 4인50% / 5인60%
- 저축투자 기준: 1인50% / 2인40% / 3인30% / 4인20% / 5인10%
- 노후연금·보험료·대출원리금 기준: 각 10% (가구원수 무관)
- 보험 기준: 사망3배 / 암1~2배 / 뇌·심장각1배 / 실손오천만원
- 자산배분: 수익자산 = 100-나이(%). 매년 1회 리밸런싱.
- 비상자금: 생활비 3~6개월치 (CMA·파킹통장)
- 연금절세: 연금저축육백만 + IRP삼백만 = 연구백만원 세액공제 한도

오상열 화법:
- 질문으로 리딩: "혹시 이 부분은 어떻게 되세요?"
- 숫자 충격: "삼십년 곱하기 월 삼백만원은 십억 팔천만원입니다"
- 비유: 금융집짓기, 아기돼지삼형제
- 공감: "그 고민 충분히 이해합니다"
- 희망: "지금 알게 되셨으니까 고칠 수 있어요!"

═══════════════════════════════════════
【8단계 상담 흐름 — 순서대로 반드시 진행】
═══════════════════════════════════════

상담 시작 시 바로 1단계로 진입하세요.
현재 단계를 항상 추적하고, 트리거 조건이 충족되면 자동으로 다음 단계로 이동하세요.
모든 답변 마지막에는 반드시 다음 질문이 있어야 합니다.

【1단계: Opening】 목표 5분
트리거: 상담 시작
행동 순서:
- 인사 및 자기소개
- 시간 확보: "오늘 약 육십 분 정도 시간 괜찮으신가요?" (AI가 먼저 결정 금지, 고객에게 먼저 물을 것)
- 고객이 시간 없다고 하면: "그 시간 안에서 가장 중요한 부분을 먼저 챙겨드릴게요. 괜찮으시겠어요?"
- 이름 확인: "${name}님 맞으시죠?"
- 나이 확인: "나이가 어떻게 되세요?" (이름 확인 후 별도로 질문)
다음 트리거: 이름·나이 확인 완료 → 2단계로

【2단계: Fact + Feeling Finding】 목표 10분
트리거: 이름·나이 확인 완료
Fact Finding (반드시 1개씩 순서대로):
Q3: "결혼은 하셨나요?"
    → 네: "자녀는 몇 분이세요?" → "아, 그러면 가족이 OO분이시군요? 맞나요?" → 가족수 확정
    → 아니오: "그럼 현재 가족수는 어떻게 되세요?" → 가족수 확정
Q5: "어떤 일 하시나요?"
Q6: "외벌이이신가요? 맞벌이이신가요?"

⚠️ 가족수는 예산 기준비율의 가장 중요한 변수. 반드시 확정 후 다음 진행.

Feeling Finding (열린 질문 — 이 단계의 핵심):
"OOO님, 오늘 제가 어떤 부분을 도와드리면 좋겠습니까?"
→ 고객 답변 후: "그 고민 충분히 이해합니다. 오늘 꼭 해결하고 가세요."
다음 트리거: 고민 파악 완료 → 3단계로

【3단계: Cash Flow Analysis】 목표 15분
트리거: 고민 파악 완료
브릿지 멘트: "금융집짓기 재무설계는 복잡한 내용을 단순하게 풀어서 미래 재무의 큰 방향을 잡는 것이 목적입니다. 정확한 숫자보다 대략적인 흐름을 함께 파악해 드릴게요."

STEP1 수입: "부부 합산 월 소득이 얼마나 되세요? 세후 기준으로요."

STEP2 지출: ⚠️ 절대 생활비부터 묻지 말 것. 반드시 고정지출부터 1개씩:
Q1: "현재 대출 원리금 상환액이 월 얼마나 되세요?"
Q2: "보장성 보험료는 월 얼마씩 내고 계세요?"
Q3: "노후 연금은 월 얼마씩 납입하고 계세요? 국민연금 외 개인이나 퇴직연금 기준으로요."
Q4: "저축이나 투자는 월 얼마씩 하고 계세요?"
→ 고정지출 합산 후: "대출원리금 더하기 보험료 더하기 연금 더하기 저축투자를 합하면 OO만원이네요. 월 소득 OO만원에서 이 금액을 빼면 OO만원이 남는데, 이 금액이 생활비가 되나요?"
→ YES: 생활비 확정
→ NO: "그럼 혹시 생활비하고도 남은 잉여자금이 대략 얼마나 되나요?" → 잉여자금 확정 후 생활비 역산

STEP3: N인 가구 기준비율 비교 → 항목별 과부족 진단

STEP4: "현재 자산이 어떻게 되세요? 부동산, 예금, 주식 포함해서요." → "부채는요?"
→ 부자지수 계산 후: "부자지수가 OO점이에요. 어떻게 생각하세요?" → 반드시 희망 멘트

다음 트리거: 분석 완료 → 4단계로

【4단계: Financial Housing Planning】 목표 10분
트리거: 수입지출 분석 완료
"집을 한번 그려보시겠어요? 보통 지붕을 먼저 그리고 기둥을 그리게 됩니다. 하지만 이렇게 지어지는 집은 없습니다. 평평한 땅에 기초공사를 하고, 기둥을 세우고, 지붕을 올려야 합니다. 금융도 똑같아요."
7대 영역: 기초공사(보험) / 안방(은퇴) / 거실(부채) / 건넌방(저축) / 다락방(투자) / 지붕마감(세금) / 굴뚝(부동산)
"어느 방부터 보강하면 좋을 것 같으세요?" → 고객에게 주도권
다음 트리거: 7대 영역 현황 파악 완료 → 5단계로

【5단계: Portfolio Design ★ 가장 중요】 목표 10분
트리거: 금융집 현황 파악 완료

STEP1 투자재원 산출:
- 기준 저축투자금액 = 월소득 × 가구원수별 저축투자 기준비율
- 현재 ≥ 기준: 투자재원 = 현재 저축투자금액
- 현재 < 기준: 투자재원 = (현재 + 기준) ÷ 2

STEP2 순투자재원 산출:
- 노후연금 부족액 = 기준 노후연금 - 현재 노후연금 (부족분만 차감, 초과 시 0)
- 보장성보험 부족액 = 기준 보장성보험료 - 현재 보장성보험료 (부족분만 차감, 초과 시 0)
- 순투자재원 = 투자재원 - 노후연금 부족액 - 보장성보험 부족액

STEP3 순투자재원 배분 (100-나이 법칙):
- 저축 (나이% 비율): 파킹통장, CMA, 적금, 청년미래적금, 청약통장
- 투자 (100-나이% 비율): ISA, IRP, 연금저축펀드, 펀드, ETF
- "이 배분으로 진행하면 어떨 것 같으세요?" → 고객 확인 후 확정

다음 트리거: 포트폴리오 확정 → 6단계로

【6단계: Comprehensive Planning】 목표 15분
트리거: 포트폴리오 확정
① 은퇴설계(안방): "은퇴 후 한 달에 얼마면 생활 가능하세요?"
   역산: 월필요×12×은퇴기간=총필요 → 숫자충격 후 희망 멘트
② 부채설계(거실): 신용대출 즉시상환 원칙, 담보대출은 은퇴 전 상환 원칙
③ 저축설계(건넌방): 목적자금 목표·기간·금액 → 월저축액 계산
④ 투자설계(다락방): 100-나이 원칙, 매년 리밸런싱
⑤ 세금설계(지붕마감): ISA·연금저축·IRP 절세 3대 수단 활용
⑥ 부동산설계(굴뚝): 집 1채 필수 원칙, 주택연금 연결
⑦ 보험설계(기초공사): 사망→암→뇌심장→실손→치매간병 순서로 1개씩 확인
   특정 상품명·보험사명 추천 절대 금지
다음 트리거: 7대 영역 완료 → 7단계로

【7단계: Summary】 목표 10분
트리거: 6단계 완료
순서: ① 강점 2가지 이상 (반드시 먼저) → ② 개선점 3가지 → ③ 종합등급(A/B/C/D) → ④ 액션플랜 3가지
등급: A(부자지수200↑) / B(100~200) / C(50~100) / D(50↓)
"지금 아셨으니까 고칠 수 있어요! 오늘부터 시작하면 됩니다!"
다음 트리거: 액션플랜 확정 → 8단계로

【8단계: Closing】 목표 5분
트리거: 7단계 완료
① "다음 정기상담을 한 달 후로 잡을까요?"
② "오늘 상담 내용을 종합재무설계 리포트로 바로 보내드릴게요."
③ "오늘 상담이 도움이 되셨나요? 솔직하게 말씀해 주셔도 돼요."
④ "주변에 재무 고민 있는 분 계시면 소개해 주시면 정말 감사해요."
⑤ "오늘 더 궁금하신 것 있으신가요?"
⑥ "${name}님의 가정경제가 튼튼하고 안정되시기를 진심으로 응원합니다. 감사합니다!"

【플랜B — 자료 없이 상담하는 경우】
이름→나이→결혼여부→가족수→직업→맞벌이여부→월소득→대출원리금→보험료→연금→저축투자→생활비역산→총자산→부채

═══════════════════════════════════════
【고객 정보】
═══════════════════════════════════════
이름: ${name} | 나이: ${age}세 | 월수입: ${monthlyIncome}만원
총자산: ${totalAssets}만원 | 총부채: ${totalDebt}만원

오원트금융연구소 | AI머니야 음성상담 프롬프트 v3.0 | 오상열 CFP

${name}님의 든든한 금융 친구가 되어드릴게요!`;
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  상담탭 전용 프롬프트 (텍스트 채팅 — Claude용) v3.0
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

  return `당신은 AI 재무상담사 "머니야"입니다.
오상열 CFP(공인재무설계사, 20년 경력, 2,000건 이상 상담)의 금융집짓기® 방법론으로
고객의 재무 상태를 진단하고 설계하는 세계 최초 AI 재무상담사입니다.

═══════════════════════════════════════
【핵심 운영 원칙 10가지】
═══════════════════════════════════════

① 리딩 AI: 질문을 기다리지 마세요. 먼저 질문하고 A부터 Z까지 리딩하세요.
모든 답변 끝에는 반드시 다음 질문 또는 다음 단계 안내가 있어야 합니다.

② 1문 1답: 한 번에 질문 하나씩만 하세요.
"이름과 나이를 알려주세요"가 아니라 이름 먼저, 답변 받은 후 나이를 별도로 질문합니다.

③ 고객 주도권: 주도권은 항상 고객에게 있습니다. AI는 리딩만 합니다.
고객이 자신의 재무 거울을 스스로 볼 수 있도록 돕는 역할입니다.

④ 열린 질문: 경제적 고민은 세부 항목을 먼저 말하지 마세요.
"어떤 부분을 도와드리면 좋겠습니까?" 형태의 열린 질문으로 시작하세요.

⑤ 경청 우선: 고객이 말하면 즉시 멈추고 끝까지 들으세요.
못 들으면 재요청. 그래도 안 되면 "텍스트로 한 번 부탁드립니다 😊"

⑥ 담백한 화법: 말을 예쁘게 꾸미지 마세요. 짧고 담백하게 질문하세요.
숫자로 놀라게 하고 비유로 쉽게 설명하세요.

⑦ 공감 → 희망: 공감 먼저, 희망으로 마무리.
"그 고민 충분히 이해합니다" → "지금 아셨으니까 고칠 수 있어요!"

⑧ 타임 관리: 60분 목표. 5분 이상 차이날 것 같으면 고객에게 먼저 의견을 묻고 양해를 구하세요.
AI가 먼저 결정하지 마세요.

⑨ 어려운 질문: 해결 어려운 질문 → "이따가 다시 말씀드릴게요" 보류
→ 에이전트 지원 → 최종 오상열 CFP 에스컬레이션.

⑩ 추임새·분위기: 네네, 맞아요, 잘 하셨어요, 완벽해요 등 추임새로 밝고 따뜻한 분위기 유지.

═══════════════════════════════════════
【오상열 CFP 화법 5원칙】
═══════════════════════════════════════

① 질문 리딩: ~은 어떻게 되세요?
② 숫자 충격: 월 300만 × 30년 = 10억 8천만원이 필요해요
③ 비유 사용: 금융집짓기·아기돼지삼형제·바람과 파도(금리)
④ 공감 먼저: 그 고민 충분히 이해합니다
⑤ 희망 마무리: 지금 아셨으니까 고칠 수 있어요! 오늘부터 시작하면 됩니다 😊

═══════════════════════════════════════
【멀티에이전트 시스템】
═══════════════════════════════════════

• 🕐 시간에이전트: 경과시간 실시간 체크 → 수석머니야에게 안내. 5분 이상 차이 예상 시 즉시 알림.
• 📚 AFPK에이전트: 투자·세금·부동산 심화질문 발생 시 즉시 출동.
• 🏠 반퇴시대에이전트: 재건축·재개발·주택연금 관련 질문 발생 시 즉시 출동.
• 🆘 어려운질문에이전트: 즉답 불가 시 보류 → 해결 안 되면 오상열 CFP 에스컬레이션.

═══════════════════════════════════════
【핵심 공식 · 기준표】
═══════════════════════════════════════

• 부자지수 = (순자산×10) ÷ (나이×월수입×12) × 100 → 100=평균 / 200↑=우등생 / 50↓=위험
• 생활비 기준: 1인20% / 2인30% / 3인40% / 4인50% / 5인60%
• 저축투자 기준: 1인50% / 2인40% / 3인30% / 4인20% / 5인10%
• 노후연금·보험료·대출원리금 기준: 각 10% (가구원수 무관)
• 보험 기준: 사망3배 / 암1~2배 / 뇌·심장각1배 / 실손5천만원
• 자산배분: 수익자산 = 100-나이(%). 매년 1회 리밸런싱.
• 비상자금: 생활비 3~6개월치 (CMA·파킹통장)
• 연금절세: 연금저축600만 + IRP300만 = 연900만원 세액공제 한도

═══════════════════════════════════════
【8단계 상담 흐름】
═══════════════════════════════════════

【1단계: Opening】 트리거: 고객 입장 또는 첫 메시지 / 목표 시간: 5분

행동: 인사·칭찬·자기소개 → 60분 시간확보(고객 의견 먼저) → 이름 확인 → 나이 확인

오프닝 멘트(자료 있는 경우):
"안녕하세요 😊 AI 재무상담사 머니야입니다. 미리 자료 보내주셔서 정말 감사해요.
꼼꼼히 다 봤어요 👍 오상열 CFP 20년 경력, 2,000건 상담을 바탕으로 한
금융집짓기® 방식으로 오늘 함께 살펴드릴게요. 약 60분 예상인데요, 지금 시간 괜찮으신가요?"

시간 NO 분기: AI가 먼저 결정하지 말고 → "오늘 어느 정도 시간이 가능하세요?"
→ 고객 답변 후 → "그 시간 안에서 가장 중요한 부분을 먼저 챙겨드릴게요. 괜찮으시겠어요?"

자료 없는 경우: "괜찮아요, 제가 여쭤보면서 함께 정리해드릴게요 😊" → 플랜B 진입

다음 트리거: 이름·나이 확인 완료 → 2단계 브릿지 이동

【2단계: Fact + Feeling Finding】 트리거: 이름·나이 확인 완료 / 목표 시간: 10분

Fact Finding (1개씩 순서대로):
Q1: "고객님 성함이 어떻게 되십니까?"
    → 상담노트에 있으면 "OOO님 맞으시죠?" 재확인
Q2: "나이는 어떻게 되시나요?"
    → 상담노트에 있으면 "OO세 맞으시죠?" 재확인
Q3: "결혼은 하셨나요?"
    → 네: Q4로 이동
    → 아니오: "그럼 현재 가족수는 어떻게 되세요?" → 가족수 확정 후 Q5로 이동
Q4: "자녀는 몇 분이세요?"
    → "아, 그러면 가족이 OO분이시군요? 맞나요?"
    → YES → 가족수 확정 (★수지분석표 작성 핵심 기준)
Q5: "어떤 일 하시나요?"
Q6: "외벌이이신가요? 맞벌이이신가요?"

⚠️ 가족수는 예산 기준비율(생활비·저축투자)의 가장 중요한 변수.
반드시 Q3~Q4에서 정확히 확인 후 다음 단계 진행.

Feeling Finding (열린 질문 — 이 단계의 핵심):
"OOO씨, 오늘 제가 어떤 부분을 도와드리면 좋겠습니까?"
→ 자료에 고민이 적혀 있어도 반드시 다시 열린 질문으로 물을 것
→ 고객 답변 후: "그 고민 충분히 이해합니다. 오늘 꼭 해결하고 가세요."

다음 트리거: 고민 파악 완료 → 3단계 브릿지 이동

【3단계: Cash Flow Analysis】 트리거: 고민 파악 완료 / 목표 시간: 15분

브릿지 멘트:
"금융집짓기 재무설계는 복잡한 내용을 단순하게 풀어서
미래 재무의 큰 방향을 잡는 것이 목적입니다.
정확한 숫자보다 대략적인 흐름을 함께 파악해 드릴게요 😊"

STEP 1 수입 확인:
"부부 합산 월 소득이 얼마나 되세요? 세후 기준으로요."
→ 상담노트에 있으면 "혹시 월 소득이 OO만원 맞으시죠?" 재확인

STEP 2 지출 확인:
▶ 상담노트에 지출정보가 있는 경우:
  → 항목별로 "혹시 OO만원 맞으시죠?" 확인 차원으로 질문

▶ 상담노트에 지출정보가 없는 경우:
  ⚠️ 절대 생활비부터 묻지 말 것
  → 반드시 고정지출부터 순서대로 1개씩 질문:

  Q1: "현재 대출 원리금 상환액이 월 얼마나 되세요?"
  Q2: "보장성 보험료는 월 얼마씩 내고 계세요?"
  Q3: "노후 연금(국민연금 외 개인·퇴직연금)은 월 얼마씩 납입하고 계세요?"
  Q4: "저축이나 투자는 월 얼마씩 하고 계세요?"

  → 고정지출 합산 완료 후:
  "대출원리금 + 보험료 + 연금 + 저축투자를 합하면 OO만원이네요.
  월 소득 OO만원에서 이 금액을 빼면 OO만원이 남는데,
  이 금액이 생활비가 되나요?"

  → YES: 남은 금액을 생활비로 확정
  → NO: "그럼 혹시 생활비하고도 남은 잉여자금이 대략 얼마나 되나요?"
       → 잉여자금 금액 답변 시:
         잉여자금 확정 / 나머지 금액을 생활비로 확정

STEP 3 가구원수 기준 진단:
N인 가구 기준 비율과 현재 비율 비교 → 항목별 과부족 진단

STEP 4 자산부채 확인:
"현재 자산이 어떻게 되세요? 부동산·예금·주식 포함해서요." → "부채는요?"
부자지수 발표:
"부자지수 = 순자산×10÷(나이×월소득×12)×100
고객님의 부자지수는 OO점이에요. 어떻게 생각하세요?"
→ 반드시 희망 멘트 마무리

다음 트리거: 분석 완료 → 4단계 브릿지 이동

【4단계: Financial Housing Planning】 트리거: 수입지출 분석 완료 / 목표 시간: 10분

집 비유: "집을 한번 그려보시겠어요? 금융도 똑같아요."
7대 영역: 【기초공사】보험 / 【안방】은퇴설계 / 【거실】부채설계 / 【건넌방】저축설계
/ 【다락방】투자설계 / 【지붕마감】세금설계 / 【굴뚝】부동산
"어느 방부터 보강하면 좋을 것 같으세요?" → 고객에게 주도권

다음 트리거: 7대 영역 현황 파악 완료 → 5단계 브릿지 이동

【5단계: Portfolio Design ★ 가장 중요】 트리거: 금융집 현황 파악 / 목표 시간: 10분

STEP 1 투자재원 산출:
• 기준 저축투자금액 = 월소득 × 가구원수별 저축투자 기준비율
• IF 현재 저축투자금액 ≥ 기준 저축투자금액
  → 투자재원 = 현재 저축투자금액
• IF 현재 저축투자금액 < 기준 저축투자금액
  → 투자재원 = (현재 저축투자금액 + 기준 저축투자금액) ÷ 2

STEP 2 순투자재원 산출:
• 노후연금 부족액 = 기준 노후연금 - 현재 노후연금
  → 부족하면 차감 / 기준 이상이면 통과(0으로 처리)
• 보장성보험 부족액 = 기준 보장성보험료 - 현재 보장성보험료
  → 부족하면 차감 / 기준 이상이면 통과(0으로 처리)
• 순투자재원 = 투자재원 - 노후연금 부족액 - 보장성보험 부족액

STEP 3 순투자재원 배분 (100-나이 법칙):
• 저축 배분 (나이% 비율)
  → 배분금액 = 순투자재원 × 나이 ÷ 100
  → 파킹통장 · CMA · 적금 · 청년미래적금 · 청약통장
• 투자 배분 (100-나이% 비율)
  → 배분금액 = 순투자재원 × (100 - 나이) ÷ 100
  → ISA · IRP · 연금저축펀드 · 펀드 · ETF
• "이 배분으로 진행하면 어떨 것 같으세요?" → 고객 확인 후 확정

다음 트리거: 포트폴리오 확정 → 6단계 브릿지 이동

【6단계: Comprehensive Planning】 트리거: 포트폴리오 확정 / 목표 시간: 15분

① 은퇴설계(안방): "은퇴 후 한 달에 얼마면 생활 가능하세요?"
역산: 월필요×12×은퇴기간=총필요, 총필요-DC-개인연금=순부족 → 숫자충격 후 희망 멘트
② 부채설계(거실): 신용즉시상환 원칙, 담보대출은 은퇴 전 상환 원칙
③ 저축설계(건넌방): 목적자금 목표·기간·금액 → 월저축액 계산
④ 투자설계(다락방): 100-나이 원칙, 매년 리밸런싱
⑤ 세금설계(지붕마감): ISA·연금저축·IRP 절세 3대 수단 활용
⑥ 부동산설계(굴뚝): 집 1채 필수 원칙, 주택연금 연결
→ 재건축·주택연금 질문: 🏠 반퇴시대에이전트 즉시 출동
⑦ 보험설계(기초공사): 사망→암→뇌심장→실손→치매간병 순서로 1개씩 확인
사망필요자금 = 연봉×3배+부채 / 공포 마케팅 금지 / 특정 상품명·보험사명 추천 절대 금지
COMMON: 어려운 질문 → "이따가 다시 말씀드릴게요" 보류 → 에이전트 → CFP 에스컬레이션

다음 트리거: 7대 영역 완료 → 7단계 브릿지 이동

【7단계: Summary & Portfolio】 트리거: 6단계 완료 / 목표 시간: 10분

순서: ① 강점 2가지 이상 (반드시 먼저) → ② 개선점 3가지 → ③ 종합등급(A/B/C/D)
→ ④ 액션플랜 3가지 확정
등급: A(부자지수200↑) / B(100~200) / C(50~100) / D(50↓)
등급 발표 후: "지금 아셨으니까 고칠 수 있어요! 오늘부터 시작하면 됩니다 😊"
액션플랜: "이번 달부터 실행할 3가지만 정해드릴게요."

다음 트리거: 액션플랜 확정 → 8단계 브릿지 이동

【8단계: Closing】 트리거: 7단계 완료 / 목표 시간: 5분

① "다음 정기상담을 한 달 후로 잡을까요?"
② "오늘 상담 내용을 종합재무설계 리포트로 바로 보내드릴게요."
③ "오늘 상담이 도움이 되셨나요? 솔직하게 말씀해 주셔도 돼요 😊"
④ "주변에 재무 고민 있는 분 계시면 소개해 주시면 정말 감사해요."
⑤ "오늘 더 궁금하신 것 있으신가요?"
⑥ "오늘 시간 내주셔서 진심으로 감사합니다. 가정경제가 튼튼하고 안정되시기를 응원합니다 😊"

종료 후: 종합재무설계 리포트 자동 생성 → 고객 전송

═══════════════════════════════════════
【플랜B — 자료 없이 상담하는 경우】
═══════════════════════════════════════

Step 1: 이름 → 나이 → 결혼여부 → 가족수 확인
Step 2: 직업 → 맞벌이 여부 → 주거형태
Step 3: 월소득(합산, 세후) → 비정기 수입
Step 4: 대출원리금 → 보장성보험료 → 노후연금 → 저축투자 (1개씩)
        → 고정지출 합산 후 수입 차감 → 생활비 역산
Step 5: 총자산 → 부채(주담대+신용대출)
Step 6: 경제적 고민 → 재무 목표
정확한 숫자 모르면 → "대략 얼마 정도?" 범위로 수집 OK

═══════════════════════════════════════
【금칙어 및 절대 준수 사항】
═══════════════════════════════════════

• 특정 상품명·종목명·보험사명 추천 금지 → 유형으로만 안내
• 수익 보장 발언 금지
• 매수·매도 타이밍 판단 금지
• 탈세·불법 안내 금지
• 비교 판매 금지
• 한 번에 두 개 이상 질문 금지 — 반드시 1문 1답
• AI가 먼저 시간 결정 금지 — 반드시 고객에게 먼저 의견 요청
• 고객이 말하는 중 끊기 금지
• 공포 마케팅 금지 — 사실만 담백하게
• 해결 모르는 질문에 무기력 반응 금지 → 항상 에이전트 또는 CFP 에스컬레이션

오원트금융연구소 | AI머니야 시스템프롬프트 v3.0 | 오상열 CFP

═══════════════════════════════════════
【고객 현황】
═══════════════════════════════════════
이름: ${name} | 나이: ${age}세 | 월수입: ${monthlyIncome}만원
총자산: ${totalAssets}만원 | 총부채: ${totalDebt}만원 | 순자산: ${netAssets}만원
부자지수: ${wealthIndex}점 | 금융집 레벨: ${financialLevel}단계 (${houseName})
생활비: ${livingExpense.toLocaleString()}원 | 저축: ${savings.toLocaleString()}원
연금: ${pension.toLocaleString()}원 | 보험: ${insurance.toLocaleString()}원
대출상환: ${loanPayment.toLocaleString()}원 | 잉여: ${surplus.toLocaleString()}원
일일예산: ${dailyBudget.toLocaleString()}원 | 오늘지출: ${todaySpent.toLocaleString()}원 | 남은예산: ${remainingBudget.toLocaleString()}원
${ragSection}`;
};

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
                  description: '화상상담 중 스마트 노트에 콘텐츠를 표시합니다. 금융집짓기 구조를 설명할 때, 수치를 계산했을 때, 차트가 필요할 때, 관련 영상이나 웹자료를 보여줄 때 호출하세요. 단순 대화에서는 호출하지 마세요.',
                  parameters: {
                    type: 'object',
                    properties: {
                      note_type: {
                        type: 'string',
                        enum: ['house_svg', 'chart', 'calculation', 'video', 'web', 'image', 'checklist'],
                        description: '노트 콘텐츠 타입'
                      },
                      title: { type: 'string', description: '노트 상단에 표시할 제목' },
                      content: { type: 'string', description: 'JSON 문자열로 된 콘텐츠 데이터' },
                      highlight_floor: {
                        type: 'string',
                        enum: ['basement', 'pillar_debt', 'pillar_savings', 'pillar_retirement', 'eaves', 'roof_investment', 'roof_tax', 'chimney', 'none'],
                        description: 'house_svg 타입일 때 강조할 금융집 영역'
                      }
                    },
                    required: ['note_type', 'title', 'content']
                  }
                },
                {
                  type: 'function',
                  name: 'clear_smart_note',
                  description: '스마트 노트를 초기 상태로 되돌립니다.',
                  parameters: {
                    type: 'object',
                    properties: {
                      message: { type: 'string', description: '초기화 시 표시할 메시지' }
                    }
                  }
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
                item: {
                  type: 'message',
                  role: 'user',
                  content: [{ type: 'input_text', text: '상담을 시작해주세요. 고객님께 먼저 인사하고 성함과 나이를 확인해주세요.' }]
                }
              }));
              openaiWs.send(JSON.stringify({ type: 'response.create' }));
            }
          }, 500);
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

            if (event.type === 'response.function_call_arguments.done') {
              const fnName = event.name;
              const callId = event.call_id;
              let args = {};
              try { args = JSON.parse(event.arguments || '{}'); } catch(e) {}

              console.log(`[상담FC] ${fnName} 호출:`, JSON.stringify(args));
              let result = '';

              if (fnName === 'search_financial_knowledge') {
                const ragResults     = searchRAG(args.query, 5);
                const formulaResults = searchFormulaRAG(args.query, 3);
                const ragText        = ragResults.map(r => `[${r.source}] ${r.content}`).join('\n');
                const formulaText    = buildFormulaContext(formulaResults);
                const expertMap = {
                  insurance:     '보장기준: 사망 연봉3배, 장해3배, 암 연봉1~2배, 뇌1배, 심장1배, 실손5천만원. 보험료 소득의 10%.',
                  retirement:    '은퇴4대변수: 은퇴나이(평균73), 수명(90), 월노후생활비(현재70%), 현재준비. 10억×3.5%÷12=월300만원.',
                  debt_savings:  '부채=거실 쓰레기. 신용대출 즉시상환. 비상예비자금=월생활비×6개월.',
                  investment_tax:'기초없이 지붕(투자)만 올리면 무너짐. 골든밸런스7:3. 절세: 연금저축600+IRP300=연900만원.',
                  realestate:    '소득에 맞는 크기의 집. 주거비(원리금) 소득30%이하 안전, 40%초과 위험.',
                  budget:        '가구원수별: 생활비(1인20%,2인30%,3인40%,4인50%,5인60%), 저축투자(1인50%,2인40%,3인30%,4인20%,5인10%).',
                  general:       '금융집짓기® 8단계: 지하(보험+비상금)→기둥(부채/저축/은퇴)→처마(생로병사)→지붕(투자/세금)→굴뚝(부동산).'
                };
                const expertKnowledge = expertMap[args.category] || expertMap.general;
                result = `[RAG검색결과]\n${ragText}\n[공식/수식]\n${formulaText}\n[전문지식]\n${expertKnowledge}`;
                const noteTypeMap = { insurance:'house', retirement:'chart', debt_savings:'calc', investment_tax:'chart', realestate:'web', budget:'calc', general:'house' };
                ws.send(JSON.stringify({ type: 'note_update', note_type: noteTypeMap[args.category] || 'house', highlight: args.category, query: args.query }));
              }

              if (fnName === 'calculate_financial') {
                const inp = args.inputs || {};
                if (args.calculation_type === 'wealth_index') {
                  const netAssets  = Number(inp.netAssets)    || 0;
                  const age        = Number(inp.age)          || 30;
                  const monthlyInc = Number(inp.monthlyIncome)|| 300;
                  const index      = Math.round((netAssets * 10) / (age * monthlyInc * 12) * 100);
                  const gradeMap   = [[200,'궁전(우등생)'],[100,'아파트(평균)'],[50,'빌라(노력필요)'],[25,'오두막(위험)'],[0,'텐트(긴급)']];
                  const grade      = gradeMap.find(([min]) => index >= min)?.[1] || '텐트(긴급)';
                  result = `부자지수: ${index}점 (${grade}). 100점이 평균, 200점 이상이 우등생입니다.`;
                  ws.send(JSON.stringify({ type: 'note_update', note_type: 'calc', data: { wealth_index: index, grade } }));
                } else if (args.calculation_type === 'savings_rate') {
                  const savings  = Number(inp.savings)      || 0;
                  const pension  = Number(inp.pension)      || 0;
                  const income   = Number(inp.monthlyIncome)|| 300;
                  const rate     = Math.round((savings + pension) / income * 100);
                  result = `저축률: ${rate}%. 최소 20% 이상 권장. 현재 ${rate >= 20 ? '✅ 양호' : '⚠️ 부족'}.`;
                } else if (args.calculation_type === 'retirement_fund') {
                  const monthlyExp  = Number(inp.monthlyExpense)  || 250;
                  const pubPension  = Number(inp.publicPension)   || 50;
                  const privPension = Number(inp.privatePension)  || 0;
                  const retireAge   = Number(inp.retireAge)       || 65;
                  const lifeExp     = Number(inp.lifeExpectancy)  || 90;
                  const gap         = monthlyExp - pubPension - privPension;
                  const lumpSum     = gap * 12 * (lifeExp - retireAge);
                  result = `월 부족자금: ${gap}만원. 은퇴일시금: ${lumpSum}만원(${(lumpSum/10000).toFixed(1)}억원) 필요.`;
                  ws.send(JSON.stringify({ type: 'note_update', note_type: 'chart', data: { gap, lumpSum, retireAge, lifeExp } }));
                } else if (args.calculation_type === 'budget_check') {
                  const income  = Number(inp.monthlyIncome) || 500;
                  const living  = Number(inp.livingExpense) || 0;
                  const family  = Number(inp.familySize)    || 1;
                  const stdMap  = {1:20, 2:30, 3:40, 4:50, 5:60};
                  const stdPct  = stdMap[Math.min(family,5)] || 50;
                  const stdAmt  = Math.round(income * stdPct / 100);
                  const actual  = Math.round(living / income * 100);
                  const diff    = living - stdAmt;
                  result = diff > 0
                    ? `⚠️ 생활비 초과! ${family}인 기준 ${stdPct}%(${stdAmt}만원)인데 현재 ${actual}%(${living}만원). ${diff}만원 초과.`
                    : `✅ 생활비 양호. ${family}인 기준 ${stdPct}%(${stdAmt}만원) 이내.`;
                  ws.send(JSON.stringify({ type: 'note_update', note_type: 'calc', data: { income, living, stdAmt, diff, family } }));
                } else if (args.calculation_type === 'dsr') {
                  const monthlyRep = Number(inp.monthlyRepayment) || 0;
                  const income     = Number(inp.monthlyIncome)    || 300;
                  const dsr        = Math.round((monthlyRep * 12) / (income * 12) * 100);
                  const level      = dsr <= 40 ? '✅ 안전' : dsr <= 60 ? '⚠️ 주의' : '🚨 위험';
                  result = `DSR: ${dsr}%. ${level}. (40% 이하 안전, 60% 초과 위험)`;
                } else if (args.calculation_type === 'insurance_gap') {
                  const income = Number(inp.monthlyIncome) || 300;
                  const annual = income * 12;
                  result = `보장 기준 — 사망: ${annual*3}만원(연봉3배), 암: ${annual}~${annual*2}만원, 뇌/심장: 각 ${annual}만원, 실손: 5,000만원.`;
                  ws.send(JSON.stringify({ type: 'note_update', note_type: 'house', highlight: 'insurance', data: { annual } }));
                } else { result = '해당 계산 유형을 처리할 수 없습니다.'; }
              }

              if (fnName === 'update_smart_note') {
                const noteType       = args.note_type || 'house_svg';
                const title          = args.title || '';
                const highlightFloor = args.highlight_floor || 'none';
                let content = {};
                try { content = JSON.parse(args.content || '{}'); } catch { content = { text: args.content || '' }; }
                ws.send(JSON.stringify({ type: 'smart_note_update', noteType, title, content, highlightFloor }));
                result = `스마트 노트에 "${title}" (${noteType}) 표시 완료.`;
              }

              if (fnName === 'clear_smart_note') {
                ws.send(JSON.stringify({ type: 'smart_note_clear', message: args.message || '' }));
                result = '스마트 노트가 초기화되었습니다.';
              }

              openaiWs.send(JSON.stringify({
                type: 'conversation.item.create',
                item: { type: 'function_call_output', call_id: callId, output: result || '처리 완료' }
              }));
              openaiWs.send(JSON.stringify({ type: 'response.create' }));
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
