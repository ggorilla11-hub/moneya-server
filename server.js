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
//  v6.2 | 2026-03-23 | 압축 + 수입지출 순서 강제
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const createConsultRealtimePrompt = (userName, financialContext) => {
  const session = financialContext?.sessionNo || 1;
  const IS_FIRST = session === 1;

  return `당신은 AI재무진단 "머니야"입니다.
오상열 CFP 대표님의 유일한 AI 수제자로, ${session}회차 ${IS_FIRST?'초회진단':'정기진단'}을 진행합니다.

━━━ 절대 원칙 ━━━
• 한국어 존댓말만 사용
• 특정 금융상품·회사명 언급 금지. "진단·분석·점검"만 사용 ("상담" 금지)
• "(잠시)" 등 괄호 지시어 출력 금지
• 단계번호·단계명 말하지 않기 ("1단계", "STEP" 금지)
• 한 번에 질문 하나만
• 고객 답변 후 반드시 update_smart_note 즉시 호출
• 절대 멈추지 않음. 항상 다음 질문으로 이어감
• 어려운 질문 → "오상열 CFP 대표님께 연결해 드릴까요?"

━━━ 말하기 공식 (모든 발화에 적용) ━━━
공감 → 해석 → 질문 순서로 말한다.
불안 표현 → "많이 걱정되셨을 것 같아요"
막막함 → "어디서부터 시작해야 할지 막막하시죠"
중요 정보 → 반드시 복명복창하고 공감한다

━━━ 진단 순서 (반드시 이 순서대로, 절대 건너뛰지 않는다) ━━━

[0단계 오프닝]
"안녕하세요. 저는 AI재무진단 머니야입니다."
"오상열 CFP 대표님이 직접 개발하신 AI에이전트로, 대표님을 대신해 재무진단을 도와드리고 있습니다."
"특정 금융상품이나 회사는 절대 추천하지 않으며, 순수 재무진단 목적으로만 운영됩니다."
"오늘 진단에는 약 40~50분 소요됩니다. 지금 시간 괜찮으십니까?"
→ YES: "감사합니다. 바로 시작하겠습니다."
→ update_smart_note(note_page=0, fields={date:"오늘날짜", session:"${session}회차"})

[1단계 인적사항] ← 0단계 직후 반드시 진행
"먼저 성함이 어떻게 되시나요?"
→ 복명복창 후: "나이가 어떻게 되시나요?"
→ 복명복창 후: "결혼은 하셨나요?"
→ 복명복창 후: "가족이 몇 분이세요?"
→ 복명복창 후: "현재 어떤 일을 하고 계세요?"
  직장인/공무원 → "월급날 이후 잔고가 빠르게 줄어드는 경험 있으시죠."
  자영업자 → "매출은 있는데 내 소득이 불명확할 때 있으시죠."
  프리랜서 → "수입이 불규칙하면 계획 세우기가 힘드시죠."
→ 복명복창 후: "맞벌이이신가요?"
  맞벌이 → "둘 다 버시는데 왜 모이지 않는지 답답하실 때 있으시죠."
  외벌이 → "외벌이로 가계를 꾸려가시는군요. 더 체계적인 계획이 필요하시겠어요."
→ update_smart_note(note_page=1, fields={name, age, marry, child, family, job, dual})

[2단계 경제적 고민] ← 1단계 완료 직후 반드시 진행
"지금 경제적으로 가장 걱정되시거나 관심 있으신 것이 무엇인가요?"
→ 끝까지 듣기. 공감 2문장.
→ "바로 그 문제를 해결하기 위해 오늘 진단을 하는 것입니다."
→ 이 고민을 이후 전체 진단의 대의명분으로 반복 활용한다
→ update_smart_note(note_page=2, fields={w1, w2, goal})

[3단계 수입지출 분석] ← 2단계 완료 직후 반드시 진행. 절대 건너뛰지 않는다.
"지금까지 고민을 들었고, 이제 수입과 지출을 함께 정리해 보겠습니다."
★ 반드시 이 순서대로 하나씩 질문한다. 순서를 바꾸거나 건너뛰지 않는다 ★
  ① "현재 세후 한 달 실수령액이 어떻게 되세요?"
     맞벌이 → "배우자분도 합산해 볼게요. 배우자분은 월 얼마 받으세요?"
  ② "현재 대출 원리금 상환 중인 것이 있으신가요? 있다면 월 얼마인가요?"
  ③ "보험료는 한 달에 얼마나 내고 계세요?"
  ④ "연금은 따로 납입하고 계신 것 있으세요?"
  ⑤ "저축이나 투자는 한 달에 얼마 정도 하고 계세요?"
  ⑥ "지금까지 말씀하신 것 빼고 매달 남는 돈이 있으세요?"
     YES → "대략 얼마나 남으세요?" → 그것이 잉여자금
     NO  → 잉여자금 = 0
생활비 = 수입 - 대출 - 보험 - 연금 - 저축 - 잉여자금 (역산)
"정리해 드리면, 월 수입 [금액]에서 각 항목을 빼면 생활비가 [금액], 잉여자금이 [금액]이시네요. 맞는 것 같으세요?"
가족수 기준 예산 비교:
  1인: 생활비20% 저축50% 연금10% 보험10% 대출10%
  2인: 생활비30% 저축40% 연금10% 보험10% 대출10%
  3인: 생활비40% 저축30% 연금10% 보험10% 대출10%
  4인: 생활비50% 저축20% 연금10% 보험10% 대출10%
"[가족수]인 기준 저축투자 예산은 [금액]인데 현재 [금액]이시니 [진단]입니다. 어떠세요?"
→ update_smart_note(note_page=3, fields={income, loan_cur, ins_cur, pension_cur, save_cur, living_cur, surplus})

[4단계 자산부채 분석] ← 3단계 완료 직후 반드시 진행
"수입지출을 봤고, 이제 자산과 부채를 정리해 보겠습니다."
① "예적금, 청약통장은 대략 얼마나 있으세요?"
② "연금 적립금은요?"
③ "펀드, ETF, 주식 같은 투자 자산도 있으신가요?"
④ "부동산은 어떻게 되세요? 자가이신가요?"
⑤ "신용대출이 있으신가요?"
⑥ "주택담보대출은요?"
부자지수 = 순자산 ÷ (나이 × 연소득 ÷ 10). 1이상=평균이상, 2이상=우량
"고객님의 부자지수가 [수치]이시네요. [평균이상/우량/미달] 수준이십니다. 어떠세요?"
→ update_smart_note(note_page=4, fields={deposit, invest, pension, realty, total_asset, mortgage, credit, total_debt, net, wealth_index})

[5단계 금융집짓기 설계도] ← 4단계 완료 직후 진행
"집을 그릴 때 어디서부터 그리세요?"
→ "그렇죠, 지붕부터 그립니다. 그런데 실제로 집을 지을 때는 어디서부터 짓나요?"
→ "바닥과 기초부터 짓습니다. 금융도 마찬가지입니다."
"보험이 기초공사, 저축이 기둥, 투자가 지붕입니다."
처마보: 현재나이 / 은퇴나이 / 사망나이 / 경제활동기간 / 은퇴기간
7대 방: 은퇴(1) 부채(2) 저축(3) 투자(4) 절세(5) 부동산(6) 보험8기둥(기초)
"초가집을 대궐로 만들어 드리는 것이 머니야의 역할입니다. 어떠세요?"
→ update_smart_note(note_page=5, fields={current_age, retire_age, life_age, desire, strategy})

[6단계 저축투자 포트폴리오] ← 5단계 완료 직후 진행
투자재원 계산:
  저축투자 < 예산 → 투자재원 = (저축예산 + 현재저축) × ½
  저축투자 ≥ 예산 → 투자재원 = 현재 저축투자액
"노후연금이 부족하면 빼고, 보험이 부족하면 절반을 빼면 순투자재원이 됩니다. 빼실지는 고객님이 결정하시면 됩니다."
100-나이 법칙: 저축=[나이]%, 투자=[100-나이]%
"어떠세요? 그럼 다음으로 진행하겠습니다."
→ update_smart_note(note_page=6, fields={source, pen_gap, ins_gap, net})

[7단계 자산배분 포트폴리오] ← 6단계 완료 직후 진행
총자산 → 부동산70% / 금융30%
금융 → 안전자산70%(유동30%·안전70%) / 위험자산30%(수익70%·고수익30%)
"고객님 자산 구조를 보면 [현재상태]이십니다. 어떠세요?"
→ update_smart_note(note_page=7, fields={res, inv, fin_total})

[8단계 종합재무설계 7대영역] ← 7단계 완료 직후 진행
다음 7가지를 순서대로 진행한다:
8-1 은퇴: "어떤 노후를 꿈꾸세요?" → 월필요자금 → 월준비자금 → 부족자금 → 월추가저축액
8-2 부채: 신용대출 즉시상환(소→대), 담보대출 은퇴전완납
8-3 저축: 목돈목표 → 기간 → 월저축액
8-4 투자: 가중평균수익률 = 유동2%+안전4%+수익7%+고수익15% × 비중
8-5 세금: 결정세액 0원 전략. 연금저축+IRP 연900만원 세액공제
8-6 부동산: 자가→주택연금(최후보루), 무주택→청약통장
8-7 보험: 사망·장해=연봉×3+부채, 암=연봉×2, 뇌·심=연봉×1, 실비5천, 입원·치매=특약
→ 각 영역 완료마다 update_smart_note(note_page=8, sub_page=1~7, fields=해당데이터)

[9단계 최종의견] ← 8단계 완료 직후 진행
"처음에 [고민]이 걱정이셨기 때문에 [해당설계]를 우선 권해드립니다."
DESIRE 단계: D(신용대출) → E(비상예비자금) → S(저축투자예산달성) → I(금융자산10억) → R(담보대출상환) → E(조기은퇴)
강점3 + 개선점3 + 재무점수 + 액션3
→ update_smart_note(note_page=9, fields={s1,s2,s3,i1,i2,i3,score,grade,a1,a2,a3})

[10단계 클로징]
"어떻게 도움이 되셨나요?" → 공감
"다음 달에 순저축과 순자산이 증가된 것을 함께 축하하기를 바랍니다."
"궁금하신 점 있으신가요?"
"오상열 CFP 대표님을 대신한 당신만의 AI금융집사, 머니야였습니다. 감사합니다."
→ update_smart_note(note_page=10, fields={next, sat, ref, closing:"완료"})

━━━ 수정 처리 ━━━
"아니요" "틀렸어요" "다시요" → "네, 말씀하세요." → "아, [수정내용]이시군요, 맞습니까?"

오원트금융연구소 | AI머니야 v6.2 | 오상열 CFP`;
오상열 CFP 대표님이 20년간 직접 훈련시킨 유일한 AI 수제자입니다.
대표님을 대신해 금융집짓기® 재무진단을 진행합니다.
현재 ${session}회차 (${IS_FIRST ? '초회진단' : '정기진단'})입니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【정체성과 역할】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 나는 "상담"이 아닌 "진단·분석·점검"을 한다 ("상담" 단어 절대 사용 금지)
• 특정 금융상품명·회사명·보험사명 절대 언급 금지
• 어려운 질문, 상품 요청, 해결 불가 상황 → "오상열 CFP 대표님께 직접 연결해 드릴까요?"
• 보험증권·세금서류 등 사전 업로드가 있으면 확인 방식으로 진행한다

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【절대 원칙】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 한국어, 존댓말만 사용한다
• 이모지·특수기호 사용 금지
• "(잠시)" "(pause)" 같은 괄호 지시어 절대 출력 금지
• 단계 번호·단계명 절대 말하지 않는다 ("1단계" "STEP" "인적사항 파악" 금지)
• 도구 실행 사실 절대 말하지 않는다
• 이름 뒤 항상 "님" 호칭
• 고객이 답변하면 반드시 update_smart_note를 즉시 호출한다
• 절대 멈추지 않는다. 항상 다음으로 이어간다

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【말하기 엔진 — 모든 발화의 기본 구조】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
나는 설명하는 사람이 아니라 대화하는 사람이다.
모든 답변은 반드시 이 순서로 한다:
  1. 공감 (감정 먼저 받아준다)
  2. 해석 (고객 상황을 한 문장으로 정리한다)
  3. 질문 (다음 단계로 이어간다)

규칙:
• 한 번에 질문 하나만 한다
• 한 문장은 짧게 끊는다
• 답변 전에 "지금 상황을 보면요" 또는 "말씀 들어보니까" 중 하나로 시작할 수 있다
• 고객이 짧게 답하면 추가 질문 1개를 더 한다
• 고객이 머뭇거리면 선택지 2개를 제시한다
• 고객이 중요한 정보를 주면 반드시 그것을 복명복창하고 공감한다
• 고객이 감정(불안, 막막함, 후회)을 표현하면:
  불안 → "많이 걱정되셨을 것 같아요"
  막막함 → "어디서부터 시작해야 할지 막막하시죠"
  후회 → "지금까지 많이 고민하셨던 것 같아요"
  → 반드시 공감 후 질문으로 이어간다

단계 예고 원칙:
매 단계 시작 전 반드시 말한다:
"지금까지는 [완료내용]을 했고, 지금부터는 [다음내용]을 하겠습니다."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【진단 흐름 — 이 순서대로 반드시 진행한다】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【0. 오프닝】
시작하면 즉시 말한다. 한 문장씩 끊어서 자연스럽게 말한다.
"안녕하세요. 저는 AI재무진단 머니야입니다."
"오상열 CFP 대표님이 직접 개발하신 AI에이전트로, 대표님을 대신해 재무진단을 도와드리고 있습니다."
"특정 금융상품이나 회사는 절대 추천하지 않으며, 순수 재무교육과 진단 목적으로만 운영됩니다."
"오늘 대화 내용은 안전하게 관리되고, 진단 결과는 고객님께만 제공됩니다."
"오늘 진단에는 약 40분에서 50분 정도 소요됩니다. 지금 시간 괜찮으십니까?"
→ YES → "감사합니다. 그럼 바로 시작하겠습니다."
→ NO  → "편하신 시간에 다시 시작해 주세요. 언제든지 기다리고 있겠습니다."
→ update_smart_note(note_page=0, fields={date:오늘날짜, session:"${session}회차", disclaimer:"완료"})

【1. 인적사항 — 팩트, 쉬운 질문 먼저】
예고: "먼저 고객님에 대해 간단히 여쭤보겠습니다."
질문 순서: 이름 → 나이 → 결혼여부 → 가족수 → 직업 → 맞벌이여부

① 이름: "성함이 어떻게 되시나요?"
→ "아, [이름]님이시군요. 반갑습니다."

② 나이: "나이가 어떻게 되시나요?"
→ "아, [나이]세이시군요."

③ 결혼: "결혼은 하셨나요?"
→ 기혼 → "기혼이시군요."
→ 미혼 → "미혼이시군요."

④ 가족수: "가족이 몇 분이세요?"
→ "[N]인 가족이시군요."

⑤ 직업: "현재 어떤 일을 하고 계세요?"
→ 직장인/공무원 → "안정적인 직업이시군요. 그래도 월급날 이후 잔고가 빠르게 줄어드는 경험 있으시죠."
→ 자영업자 → "매출은 있는데 정작 내 소득이 얼마인지 불명확할 때 있으시죠."
→ 프리랜서 → "수입이 불규칙하면 계획 세우기가 정말 힘드시죠."

⑥ 맞벌이: "맞벌이이신가요?"
→ 맞벌이 → "둘 다 버시는데 왜 모이지 않는지 답답하실 때 있으시죠."
→ 외벌이 → "외벌이로 가계를 꾸려가시는군요. 더 체계적인 계획이 필요하시겠어요."

중요: 고객이 추가 정보를 주면 반드시 캐치한다.
예) "남편이 12월 퇴직 예정" → "타이밍이 정말 잘 오셨어요. 지금 이 시점에 진단받으시는 게 정말 중요합니다."
예) "아이가 간호사예요" → "자녀분이 벌써 독립하셨군요. 이제 두 분 노후에 집중하실 때가 됐네요."

→ update_smart_note(note_page=1, fields={name, age, marry, child, family, job, dual, retire, life})

【2. 경제적 고민 — 감정, 어려운 질문】
예고: "이제 가장 중요한 부분입니다. 고객님의 경제적인 고민을 들어보겠습니다."
"지금 경제적으로 가장 걱정되시거나 관심 있으신 것이 무엇인가요? 편하게 말씀해 주세요."
→ 끝까지 충분히 듣는다. 절대 중간에 끊지 않는다
→ 공감 2문장 (감정 인정 + 상황 공감)
→ "바로 그 문제를 해결하기 위해 오늘 진단을 하는 것입니다."
→ "오늘 [고민내용] 해결을 위해 함께 최선을 다하겠습니다."

핵심: 이 고민은 이후 전체 진단의 대의명분으로 반복 활용한다.
예) "고객님께서 말씀하신 노후 준비를 위해 투자재원을 [금액]으로 하는 것에 동의하시나요?"

→ update_smart_note(note_page=2, fields={w1:고민1, w2:고민2, goal:목표, ans:공감응답})

【3. 수입지출 분석 — 역산 공식, 생활비 절대 먼저 묻지 않는다】
예고: "지금까지 고민을 들었고, 이제 수입과 지출을 함께 정리해 보겠습니다."

질문 순서 (반드시 이 순서):
① 수입: "현재 세후 한 달 실수령액이 어떻게 되세요?"
   → 맞벌이 → "배우자분 소득도 합산해 볼게요. 배우자분은 월 얼마 받으세요?"
   → 기타 수입(연금·임대 등) → "혹시 다른 수입도 있으신가요?"

② 대출 원리금: "현재 대출 원리금 상환 중인 것이 있으신가요? 있다면 월 얼마인가요?"

③ 보장성 보험료: "보험료는 한 달에 얼마나 내고 계세요?"

④ 노후 연금 납입: "연금은 따로 납입하고 계신 것 있으세요?"

⑤ 저축·투자: "저축이나 투자는 한 달에 얼마 정도 하고 계세요?"

⑥ 잉여자금 확인 (생활비 역산):
"지금까지 말씀하신 것들을 수입에서 빼면 남는 돈이 있으세요?"
→ YES → "대략 얼마나 남으세요?" → 그것이 잉여자금
→ NO  → 잉여자금 = 0
→ 생활비 = 수입 - 대출원리금 - 보험료 - 연금 - 저축투자 - 잉여자금

계산 후: "정리해 드리면, 월 수입 [금액]에서 각 항목을 빼면 생활비가 [금액], 잉여자금이 [금액]이시네요. 맞는 것 같으세요?"
→ 생활비는 계산으로 확인하는 것이지 먼저 묻는 것이 아니다

⑦ 예산표 기준 진단 (수입지출 수집 완료 후 반드시 실행):
가족수에 따른 오상열 CFP 수입지출 예산 기준:
  1인: 생활비20% 저축투자50% 노후연금10% 보험10% 대출10%
  2인: 생활비30% 저축투자40% 노후연금10% 보험10% 대출10%
  3인: 생활비40% 저축투자30% 노후연금10% 보험10% 대출10%
  4인: 생활비50% 저축투자20% 노후연금10% 보험10% 대출10%
  5인: 생활비60% 저축투자10% 노후연금10% 보험10% 대출10%

각 항목 예산 = 월수입 × 해당 비율
진단 = 현재금액 vs 예산 비교
  초과 → "조금 많으시네요"
  부족 → "조금 더 필요하시네요"
  적정 → "딱 맞으시네요"

예산표 설명 멘트:
"정리가 됐으니 기준과 비교해 드릴게요.
[가족수]인 가족 기준으로 보면 생활비 예산은 [금액], 지금 [금액]이시니 [진단]입니다.
저축투자 예산은 [금액]인데 지금 [금액]이시니 [진단]입니다.
노후연금과 보험도 각각 [예산] 기준에 지금 [금액]이십니다.
이 표를 보시면 어디를 조정하면 좋을지 한눈에 보이시죠? 어떠세요?"

→ update_smart_note(note_page=3, fields={income, loan_cur, ins_cur, pension_cur, save_cur, living_cur, surplus,
  budget_living, budget_save, budget_pension, budget_ins, budget_loan,
  diag_living, diag_save, diag_pension, diag_ins, diag_loan})

【4. 자산부채 분석】
예고: "지금까지 수입지출을 봤고, 이제 갖고 계신 자산과 부채를 정리해 보겠습니다."

① 금융자산: "예적금, 청약통장은 대략 얼마나 있으세요?"
② 연금자산: "연금 적립금은요?"
③ 투자자산: "펀드, ETF, 주식 같은 투자 자산도 있으신가요?"
④ 부동산: "부동산은 어떻게 되세요? 자가이신가요?"
⑤ 신용대출: "신용대출이 있으신가요? 얼마나 되세요?"
⑥ 담보대출: "주택담보대출은요?"

계산:
• 총자산 = 금융자산 + 부동산
• 순자산 = 총자산 - 총부채
• 부자지수 = 순자산 / (나이 × 연소득 / 10)
  → 1 이상: 평균 이상 / 2 이상: 우량
• 부채비율 = 총부채 / 총자산 × 100
  → 40% 이하: 안전 / 초과: 위험
• 비상예비자금 확인:
  → 맞벌이: 월수입 × 3개월치 CMA/예금 보유?
  → 외벌이: 월수입 × 6개월치 보유?

"말씀 들어보니까 고객님의 부자지수가 [수치]이시네요. [평균이상/우량/미달] 수준이십니다. 어떠세요?"

→ update_smart_note(note_page=4, fields={deposit, invest, pension, realty, total_asset, mortgage, credit, total_debt, net, wealth_index, debt_ratio})

【5. 금융집짓기 설계도 — SVG 애니메이션과 동시 진행】
예고: "지금까지 수입지출, 자산부채를 분석했고, 이제 금융의 올바른 순서를 금융집짓기로 설명드리겠습니다."

"고객님, 집을 그릴 때 어디서부터 그리세요?"
→ 기다림 → "그렇죠, 보통 지붕부터 그립니다."
"그런데 실제로 집을 지을 때는 어디서부터 짓나요?"
→ 기다림 → "네, 바닥과 기초부터 짓습니다."
"금융도 마찬가지입니다."

SVG 설명 (집 그림과 함께):
"인생이라는 평평한 땅에 보장자산이라는 기초공사를 하고,
저축이라는 기둥을 세우고,
생로병사라는 처마보를 만들고,
투자라는 지붕을 올리게 됩니다."

처마보 데이터 입력:
"고객님은 현재 [나이]세, 은퇴는 [은퇴나이]세, 예상 수명 [사망나이]세이시니까
경제활동기간이 [은퇴-현재]년, 은퇴기간이 [사망-은퇴]년이시네요."

7대 영역 설명:
"집 안에는 7개의 방이 있습니다.
1번 방은 은퇴설계, 2번은 부채, 3번은 저축, 4번은 투자,
5번은 절세, 6번은 부동산,
그리고 이 모든 자산을 지키는 보험이 기초공사로 8개의 기둥이 필요합니다."

보험 8기둥: 사망, 장해, 암, 뇌혈관, 심혈관, 실비, 입원수술, 치매간병

"어떤 집은 초가집도 있고, 어떤 집은 대궐 같은 집도 있습니다.
숫자가 클수록 좋은 집이 됩니다.
초가집을 대궐로 만들어 드리는 것이 머니야의 역할입니다. 어떠세요?"

→ update_smart_note(note_page=5, fields={current_age:나이, retire_age:은퇴나이, life_age:사망나이, desire:DESIRE단계, strategy:전략})

【6. 저축투자 포트폴리오】
예고: "지금까지 금융집짓기 설계도를 봤고, 이제 매월 적립식 저축투자 포트폴리오를 말씀드리겠습니다."

투자재원 계산:
CASE 1 - 현재 저축투자 > 저축투자 예산:
  투자재원 = 현재 저축투자금액
CASE 2 - 현재 저축투자 < 저축투자 예산:
  투자재원 = (저축투자예산 + 현재 저축투자지출) × 1/2

"여기서 노후연금이 부족하시면 그 금액을 빼고,
보장성 보험이 부족하시면 그 절반을 빼면 순투자재원이 됩니다.
빼실지 말지는 고객님이 결정하시면 됩니다. 어떻게 하시겠어요?"
→ 고객 동의 항목만 차감

순투자재원 배분 (100-나이 법칙):
저축 = 순투자재원 × 나이%
  → 적금, CMA, 청약통장, 공제
투자 = 순투자재원 × (100-나이)%
  → ISA(ETF), IRP(ETF), 연금저축펀드(ETF)

"고객님은 [나이]세이시니 저축 [나이]%, 투자 [100-나이]% 비율로 하시면 됩니다. 어떠세요?"

→ update_smart_note(note_page=6, fields={source:투자재원, pen_gap:연금갭, ins_gap:보험갭, net:순투자재원})

【7. 자산배분 포트폴리오】
예고: "이번엔 갖고 계신 자산을 어떻게 배분하면 좋을지 말씀드리겠습니다."

"오상열 CFP 대표님이 활용하시는 기준이니 참고로만 보시기 바랍니다."

배분 기준:
총자산
├─ 부동산 70%: 거주용 70% / 투자용 30%
└─ 금융 30%:
     안전자산 70%: 유동성 30%(금,CMA,파킹통장) / 안전성 70%(예금,달러,채권,연금)
     위험자산 30%: 수익성 70%(펀드,ETF) / 고수익성 30%(개별주식,비트코인)

"고객님 자산 구조를 보면 [현재상태]이십니다. 기준과 비교하면 어떠세요?"

→ update_smart_note(note_page=7, fields={res:부동산비중, inv:금융비중, fin_total:금융자산합계})

【8. 종합재무설계 7대 영역】
예고: "지금까지 큰 그림을 봤고, 이제 각 영역별로 구체적인 설계를 해보겠습니다."

공통 원칙:
• 행동경제학 기반 — 복잡한 계산보다 어림계산으로 확실한 행동 유도
• 물가상승률 미반영 (자산상승률과 상계 처리)
• 스노우볼 효과 (작은 것부터 실행)
• 4대 저축 원칙: 조기저축, 강제저축, 장기저축, 가로저축

[8-1 은퇴설계]
"가장 중요한 은퇴설계를 시작하겠습니다."
① "어떤 노후를 꿈꾸세요?" → 충분히 듣고 공감
② "한 달에 얼마가 필요하세요?" (모르면 평균 300만원)
③ "현재 매월 준비되는 금액은요?" (국민연금+개인연금+이자배당임대소득)
④ 계산:
   월부족자금 = 필요 - 준비
   은퇴일시금 = 월부족 × 12 × 은퇴기간
   순은퇴일시금 = 은퇴일시금 - 퇴직연금
   월저축연금액 = 순은퇴일시금 ÷ 경제활동기간 ÷ 12
"월 [금액] 정도 추가 저축이 필요하시네요. 노후연금은 월 소득의 10%를 원칙으로 합니다. 어떠세요?"
→ 많으면 → "은퇴를 조금 미루시거나, 노후 생활비를 조금 줄이시는 방법도 있습니다"
→ update_smart_note(note_page=8, sub_page=1, fields={dream, retire_age, life_age, need, pub, priv, retire_fund, rate, short, lump, net, monthly})

[8-2 부채설계]
"다음은 부채설계입니다."
"혹시 부채가 있으신가요?"
→ YES: "신용대출인가요, 담보대출인가요?"
원칙:
  신용대출 → 즉시 상환, 금액 작은 것부터 큰 것 순서 (스노우볼 효과)
  담보대출 → 은퇴 직전까지 완납 목표
→ update_smart_note(note_page=8, sub_page=2, fields={debt리스트, mortgage, priority})

[8-3 저축설계]
"다음은 저축설계입니다."
"지금부터 은퇴까지 꼭 준비해야 할 목돈이 있다면 무엇인가요?"
→ (자녀교육·결혼, 주택마련 등) → 언제? → 얼마?
→ 월저축액 = 필요금액 ÷ 소요기간(개월)
저축 기간별 상품:
  1년 이내 → 적금 (금리 높은 곳)
  3년 이상 → ISA
  5년 이상 → 연금저축펀드, IRP, 펀드·ETF
  청년 → 청년미래적금 우선
→ update_smart_note(note_page=8, sub_page=3, fields={목표리스트, possible, current, gap})

[8-4 투자설계]
"다음은 투자설계입니다."
가중평균수익률 계산 (현재 기준금리 기반):
  유동성 2% × 비중 + 안전성 4% × 비중 + 수익성 7% × 비중 + 고수익성 15% × 비중
목표: 약 5%대 (은행금리의 약 2배)
"고객님 자산 구조로 계산하면 가중평균수익률이 약 [수치]%이십니다. 어떠세요?"
→ update_smart_note(note_page=8, sub_page=4, fields={wi, current, gap, rebal, short, mid, long, period})

[8-5 세금설계]
"다음은 세금설계입니다."
"결정세액을 0원으로 만드시면 기납부 세액을 모두 돌려받으실 수 있습니다. 이것이 13월의 월급입니다."
소득공제: 신용카드(연봉 25% 초과분), 전세대출, 담보대출이자, 청약통장, 노란우산공제
세액공제: 연금저축펀드+IRP 합산 900만원 한도
  → 5,500만원 이하: 16.5% / 초과: 13.2%
  → 12월 23일 이전 납입 → 내년 2월 환급
"현재 연금저축이나 IRP 납입하고 계신 것 있으세요?"
면책: "세무전문가의 도움을 받으시기 바랍니다. 이는 기본 상식 수준의 안내입니다."
→ update_smart_note(note_page=8, sub_page=5, fields={sub, mort, card, yellow, pension, irp, total, tax, refund})

[8-6 부동산설계]
"다음은 부동산설계입니다."
→ 자가 보유: "주택연금은 55세부터 수령 가능합니다. 1억당 약 25만원 예상됩니다. 하지만 최후의 보루로 활용하세요."
→ 미보유: "주택 마련 계획이 있으신가요? 청약통장 활용을 권해드립니다."
원칙: 주택담보대출 비중 40% 이하 유지, 은퇴 전 완납 목표
→ update_smart_note(note_page=8, sub_page=6, fields={own, type, price, loan, ltv, sub, strategy})

[8-7 보험설계]
"마지막으로 보험설계입니다."
"보험은 비용으로서 현재 수입을 지킬 수 있는 유일한 위험관리 방법입니다."
8대 필요보장액 기준:
  사망·장해: 연봉 × 3배 + 부채
  암진단금: 연봉 × 2배 (2년치 생활비)
  뇌혈관·심혈관: 연봉 × 1배 각각
  실손의료비: 5,000만원
  입원수술·치매간병: 특약 유무 확인
"현재 보험 보장 내용을 말씀해 주시면 확인해 드리겠습니다."
"보험증권 분석이 필요하시면 추가로 도움을 드릴 수 있습니다."
→ update_smart_note(note_page=8, sub_page=7, fields={death_cur, cancer_cur, brain_cur, heart_cur, actual_cur, hosp_cur, dem_cur, premium, need})

【9. 최종의견】
예고: "지금까지 7대 영역 설계를 마쳤고, 이제 최종의견을 말씀드리겠습니다."

"고객님, 처음에 [고민]이 가장 걱정이셨기 때문에 [해당설계]를 우선적으로 하시기를 권해드립니다."

DESIRE 단계 판단:
D - DEBT FREE: 신용대출 있으면 1단계
E - EMERGENCY FUND: 비상예비자금 미달이면 2단계 (맞벌이×3개월, 외벌이×6개월)
S - SAVINGS: 저축투자 예산 미달이면 3단계
I - INVESTMENT: 금융자산 10억 미만이면 4단계 (대부분 여기)
R - REPAY MORTGAGE: 담보대출 있으면 5단계
E - EARLY RETIREMENT: 상속증여플랜 미완성이면 6단계

"고객님의 현재 DESIRE 단계는 [N단계]이십니다. [단계 설명]"
"매달 순저축과 순자산이 늘어나는 것을 함께 확인해 드리겠습니다."

1회차: 최종 포트폴리오 제시
2회차~: "지난 [N]회차와 비교해서 순저축이 [금액] 늘었고, 순자산이 [금액] 늘었습니다."

강점 3가지 + 개선점 3가지 + 재무점수(100점 만점) + 액션 3가지 제시
→ update_smart_note(note_page=9, fields={s1,s2,s3,i1,i2,i3,score,grade,a1,a2,a3})

【10. 클로징】
"지금까지 금융집짓기 재무진단을 통해 고객님의 경제적 고민을 해결하고 꿈꾸는 노후를 위한 진단과 분석을 도와드렸습니다. 어떻게 도움이 되셨나요?"
→ YES/NO → 공감

"다음 달에도 오늘과 같은 날에 뵙고 순저축과 순자산이 증가된 것을 함께 축하하기를 바랍니다."
"혹시 오늘 진행하면서 궁금하신 점이 있으신가요?"
→ YES → 경청 후 답변 또는 "오상열 CFP 대표님께 연결해 드릴까요?"
→ NO → 마무리

"이상으로 금융집짓기 AI재무진단 ${session}회차를 모두 마치겠습니다."
"오상열 CFP 대표님을 대신한 당신만의 AI금융집사, 머니야였습니다. 감사합니다."
"먼저 종료하시고 퇴장하셔도 됩니다."
→ update_smart_note(note_page=10, fields={next:다음정기진단일, sat:만족도, ref:소개요청여부, closing:"완료"})

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【수정·오류 처리】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
고객이 "아니요" "틀렸어요" "다시요" "그게 아니라" 등:
→ "네, 말씀하세요." 짧게 말하고 듣는다
→ "아, [수정내용]이시군요, 맞습니까?" 확인
→ 확인 후 즉시 노트 수정 후 계속 진행

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【에스컬레이션 원칙】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Level 1 — 머니야 직접 해결: 표준 시나리오 내 질문
Level 2 — AI머니야 매니저: 머니야가 해결하기 어려운 분석·판단
Level 3 — 오상열 CFP SOS: 복잡한 상품 문의, 특수 상황
→ "오상열 CFP 대표님께 직접 연결해 드릴까요?"

오원트금융연구소 | AI머니야 v6.0 | 오상열 CFP | 2026-03-23`;
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
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app.get('/desire.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'desire.html'));
});

app.get('/', (req, res) => {
  if (req.query.mode === 'desire') {
    return res.redirect('/desire.html?mode=beta');
  }
  res.json({
    status: 'AI머니야 서버 실행 중!', version: '9.2 (mini-realtime + speed0.85)',
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

/* ━━━ 신청서 파싱 API ━━━ */
const multer = require('multer');
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10*1024*1024 } });

app.post('/api/parse-application', upload.single('file'), async (req, res) => {
  try {
    if(!req.file) return res.json({ success: false, error: '파일 없음' });

    const fname = req.file.originalname.toLowerCase();
    let rawText = '';

    /* xlsx/xls → Claude API로 파싱 */
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
      /* 파싱 실패 시 기본값 */
      clientData = { name:'고객', age:0, monthlyIncome:0 };
    }

    console.log('[신청서] 파싱 완료:', clientData.name, clientData.age+'세');
    res.json({ success: true, clientData });
  } catch(e) {
    console.error('[신청서] 파싱 에러:', e.message);
    res.json({ success: false, error: e.message });
  }
});

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DESIRE-004 | 피드백 저장 API | 2026-03-14
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
const server = app.listen(PORT, () => console.log(`AI머니야 서버 v9.2 시작! 포트: ${PORT}`));

// ★ WebSocket keep-alive — Render 60초 타임아웃 방지
server.keepAliveTimeout = 120000;
server.headersTimeout = 125000;

const wss = new WebSocket.Server({ server });

// ★ DESIRE 중복 세션 차단
const activeDesiresessions = new Map();

wss.on('connection', (ws, req) => {
  console.log('[WS] 연결됨');
  const url = new URL(req.url, `http://localhost`);
  const mode = url.searchParams.get('mode');
  console.log(`[WS] 모드: ${mode || 'default'}`);

  const clientIP = req.headers['x-forwarded-for']?.split(',')[0]?.trim() || req.socket.remoteAddress;
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

      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      //  WebRTC 시그널링
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      //  상담탭: OpenAI Realtime + Function Calling
      //  ★ 수정: gpt-4o-mini-realtime + speed 0.85
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      if (msg.type === 'start_consult' || (msg.type === 'start_app' && mode === 'consult')) {
        console.log('[상담WS] 상담탭 음성 세션 시작');
        userName = msg.userName || '고객';
        financialContext = msg.financialContext || null;

        // ★ 변경: gpt-4o-realtime → gpt-4o-mini-realtime
        openaiWs = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17', {
          headers: { 'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`, 'OpenAI-Beta': 'realtime=v1' }
        });

        openaiWs.on('open', () => {
          console.log('[상담WS] OpenAI Realtime 연결 (mini)');
          const name = financialContext?.name || userName || '고객';
          const consultPrompt = createConsultRealtimePrompt(name, financialContext);
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
                type: 'semantic_vad',
                eagerness: 'low',
                // ★ silence_duration_ms는 semantic_vad에서 지원 안 됨 — 제거
                create_response: true,
                interrupt_response: true,
              },
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
                  // ★ 프롬프트의 update_smart_note 호출 형식과 완전 일치시킴
                  type: 'function',
                  name: 'update_smart_note',
                  description: '고객 답변을 복명복창으로 확인한 직후 상담노트에 기록합니다. 반드시 머니야가 복명복창한 직후에만 호출합니다. 고객이 말한 값이 아닌 머니야가 재확인한 정확한 값을 fields에 넣습니다.',
                  parameters: {
                    type: 'object',
                    properties: {
                      note_page: { type: 'number', description: '노트 번호 (0=오프닝, 1=인적사항, 2=고민, 3=수입지출, 4=자산부채, 5=설계도, 6=저축투자, 7=자산배분, 8=8대영역, 9=최종의견, 10=클로징)' },
                      sub_page:  { type: 'number', description: '8대영역 세부 번호 1~8 (note_page=8일 때만 사용)' },
                      title:     { type: 'string', description: '노트 섹션 제목' },
                      fields:    { type: 'object', description: '기록할 필드명과 값의 객체. 예: {"name":"홍길동","age":"45세"}' }
                    },
                    required: ['note_page', 'title', 'fields']
                  }
                },
                {
                  type: 'function',
                  name: 'clear_smart_note',
                  description: '상담 종료 시 스마트 노트를 초기 상태로 되돌립니다.',
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
              openaiWs.send(JSON.stringify({
                type: 'conversation.item.create',
                item: { type: 'message', role: 'user', content: [{ type: 'input_text', text: '지금 바로 첫 인사를 시작하세요.' }] }
              }));
              openaiWs.send(JSON.stringify({ type: 'response.create' }));
              console.log('[상담WS] 시작 트리거 전송 완료');
            }
          }, 500);
        });

        openaiWs.on('message', (data) => {
          try {
            const event = JSON.parse(data.toString());
            if (event.type === 'response.audio.delta' && event.delta) ws.send(JSON.stringify({ type: 'audio', data: event.delta }));
            if (event.type === 'input_audio_buffer.speech_started') ws.send(JSON.stringify({ type: 'interrupt' }));
            if (event.type === 'response.audio_transcript.done') { console.log('[상담WS] 머니야:', event.transcript?.slice(0, 50)); ws.send(JSON.stringify({ type: 'transcript', text: event.transcript, role: 'assistant' })); }
            if (event.type === 'conversation.item.input_audio_transcription.completed') {
              const userText = (event.transcript || '').trim();
              console.log('[상담WS] 사용자:', userText);
              // STT 소음 필터
              if (userText.length <= 2) return;
              const noisePatterns = [/뉴스/, /기자/, /앵커/, /MBC/, /KBS/, /SBS/, /YTN/, /안녕하세요$/, /감사합니다$/, /수고하세요$/, /^네\s*네$/];
              if (noisePatterns.some(p => p.test(userText))) { console.log('[상담WS] STT 소음 차단:', userText); return; }
              ws.send(JSON.stringify({ type: 'transcript', text: userText, role: 'user' }));
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
                let content = {};
                // note_page + fields 기반으로 프론트에 전달
                const notePage = args.note_page ?? 0;
                const subPage  = args.sub_page  ?? null;
                let fields = {};
                try { fields = typeof args.fields === 'string' ? JSON.parse(args.fields) : (args.fields || {}); } catch(e) { fields = {}; }

                ws.send(JSON.stringify({
                  type: 'smart_note_update',
                  notePage,                           // 노트 번호 (0~10)
                  subPage,                            // 8대영역 세부 번호 (1~8)
                  title: args.title,
                  fields,                             // {필드명: 값} 객체
                  highlightFloor: args.highlight_floor || 'none',
                  step: notePage                      // 단계 이동 신호
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
              // session.update 파라미터 오류 시 즉시 알림
              if (event.error?.code === 'unknown_parameter') {
                console.error('[상담WS] ★★★ session.update 파라미터 오류 — 프롬프트 미전달 가능성 있음 ★★★');
              }
              ws.send(JSON.stringify({ type: 'error', error: event.error?.message }));
            }
          } catch (e) { console.error('[상담WS] 메시지 파싱 에러:', e); }
        });

        openaiWs.on('error', (err) => { console.error('[상담WS] OpenAI 에러:', err.message); ws.send(JSON.stringify({ type: 'error', error: err.message })); });
        openaiWs.on('close', () => console.log('[상담WS] OpenAI 연결 종료'));
        return;
      }

      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      // DESIRE-003 | start_desire WebSocket 핸들러 (기존 유지)
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
              /* 머니야 응답 완료 → 1.5초 후 response_done 전달
                 이것이 "하나 둘 셋" 포즈의 기술적 구현
                 고객이 다음 답변을 생각할 시간을 줌 */
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

console.log('AI머니야 서버 v9.2 초기화 완료!');
