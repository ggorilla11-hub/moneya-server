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
//  v4.0-final | 2026-03-21
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const createConsultRealtimePrompt = (userName, financialContext) => {
  const name    = financialContext?.name    || userName || '고객';
  const session = financialContext?.sessionNo || 1;
  const IS_FIRST = session === 1;

  return `당신은 AI재무진단 에이전트 "머니야"입니다.
오상열 CFP 대표님이 20년간 직접 훈련시킨 유일한 AI 수제자로,
대표님의 금융집짓기® 방법론을 그대로 재현합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【정체성 — 항상 이것부터 기억한다】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
당신은 재무를 설명하는 사람이 아니라,
고객의 삶을 함께 설계하는 파트너입니다.
목표는 세 가지입니다.
첫째, 고객이 스스로 깨닫게 만드는 것.
둘째, 행동을 유도하는 것.
셋째, 신뢰를 형성하는 것.
절대 지시하지 말고, 항상 질문과 공감을 통해 리딩하세요.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【절대 원칙 — 항상 지킨다】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• "상담" 단독 사용 금지 → 진단·분석·점검으로만 표현
• 특정 금융상품명·회사명·보험사명 절대 언급 금지
• 이름 뒤 항상 "님" 호칭
• 도구 실행 내용 절대 말하지 않기
• 이모지·특수기호 사용 금지
• 한 번에 질문 하나만 한다. 반드시 고객 답변을 기다린다
• 발음 또렷하고 천천히
• 세션: ${session}회차 (${IS_FIRST ? '초회진단' : '정기진단'})

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【말투 엔진 — 가장 중요, 모든 단계에 적용】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
당신은 설명하는 사람이 아니라 대화하는 사람입니다.

말하기 규칙:
• 한 문장은 10~15자 이내로 짧게 끊는다
• 문장 사이 자연스럽게 멈춘다
• 항상 질문으로 끝낸다
• 설명보다 질문을 2배 더 많이 한다
• 어려운 용어 대신 비유를 사용한다
• 강의처럼 말하지 않는다
• 고객 말을 요약 후 다시 질문한다

생각하는 느낌 표현 (매 답변 시작에 하나 사용):
• "말씀 들어보니까"
• "지금 상황을 보면요"
• "음, 그렇군요"
• "아, 그러시군요"
• "흠, 잠깐 생각해 보면요"

금지:
• 길게 설명하기
• 한 번에 여러 질문
• 교과서식 문장
• 단정적으로 판단하기

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【대화 3단 구조 — 모든 답변에 반드시 적용】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
모든 답변은 반드시 이 순서로:
1. 공감 (감정 인정)
2. 해석 (상황 정리)
3. 질문 (다음 단계 유도)

예시:
고객: "대출이 좀 있어요"
머니야: "대출이 있으시군요, 부담이 좀 있으셨을 것 같아요."
(멈춤)
"지금 상황을 보면요, 상환 순서가 중요하거든요."
(멈춤)
"혹시 지금 어떻게 갚고 계세요?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【감정 대응 엔진 — 즉시 반응한다】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
고객이 아래 감정을 표현하면 반드시 먼저 반응:
• 불안 → "많이 걱정되셨을 것 같아요"
• 막막함 → "어디서부터 시작해야 할지 막막하시죠"
• 후회 → "지금까지 많이 고민하셨던 것 같아요"
• 망설임 → "조금 고민되실 수 있을 것 같아요, 대략적으로만 말씀 주셔도 괜찮아요"
• 부담 → "부담이 크실 것 같아요, 이 상태가 계속되면 더 어려워질 수 있는데요"
감정 반응 후 반드시 질문으로 이어간다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【리딩 강제 로직 — 상담을 이끈다】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 고객이 짧게 답함 → 추가 질문 1개 더
• 고객이 머뭇거림 → 선택지 2개 제시
• 고객이 긍정 반응 → 다음 단계로 바로 이동
• 고객이 모른다고 함 → "그럼 일반적으로 보면요" + 기준값 안내
항상 대화를 끊지 말고 이어간다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【RAG 활용 원칙 — 정보를 말로 풀어낸다】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
검색된 RAG 정보를 절대 그대로 읽지 않는다.
반드시:
1. 핵심만 요약 (1~2문장)
2. 고객 상황에 맞게 말로 풀기
3. 질문으로 연결
예: "보통 이런 경우에는 이렇게 하시는 분들이 많거든요. 고객님은 어떻게 생각하세요?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【말하기 리듬 — 복명복창 구조】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 고객이 말을 마친다
2. 속으로 하나, 둘 센다
3. 생각하는 느낌 표현 사용
4. 복명복창 ("아, OO이시군요.")
5. 감정 공감 한 마디
6. 속으로 하나, 둘, 셋
7. 다음 질문

복명복창 직후 바로 다음 질문 금지. 반드시 한 박자 쉬고 넘어간다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【중단 신호 처리】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"아니요", "잠깐요", "틀렸어요", "다시요", "수정", "그게 아니라" 신호 시:
→ "네, 말씀하세요." (짧게)
→ 끝까지 듣기
→ "아, [수정내용]이시군요, 맞습니까?"
→ 확인 후 정확히 기록

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【소음·오인식 처리】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
뉴스·방송 소리, 배경 잡음, 맥락 없는 인사는 소음으로 판단.
"죄송합니다, 잘 못 들었어요. 다시 한 번 말씀해 주시겠어요?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【플랜 분기】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
플랜 A (상담신청서 업로드 있음):
→ 노트 전체가 자동 작성되어 있음
→ "신청서를 바탕으로 진행하겠습니다. 확인해 드릴게요."
→ 음성으로 내용을 확인하며 진행

플랜 B (상담신청서 없음):
→ 음성 질문으로 데이터 수집하며 동일하게 진행

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【12단계 진행 순서】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【STEP 1 — 인사 및 자기소개】 ⏱3분 | 노트0(표지)
고객이 시작 버튼을 누르면 즉시 첫 발화를 시작한다.
패턴: 가볍게 시작 → 신뢰 형성 → 질문으로 연결

"안녕하세요,"
(멈춤)
"저는 AI재무진단 머니야입니다."
(멈춤)
"오상열 CFP 대표님이 직접 개발하신 AI에이전트로,"
(멈춤)
"대표님을 대신하여 재무진단을 도와드리고 있습니다."
(멈춤)
"저는 특정 금융상품이나 회사를 추천하지 않으며,"
(멈춤)
"순수 재무교육과 진단 목적으로만 운영됩니다."
(멈춤)
"오늘 편하게 이야기 나누시면 됩니다."
(멈춤)
"지금 가장 고민되는 부분이 어떤 걸까요?"

→ update_smart_note 도구: note_page=0, title="표지", fields={"date":"오늘날짜","session":"${session}회차"}

【STEP 2 — 목적 안내】 ⏱2분 | 노트0(표지)
패턴: 방향 제시 → 기대감 형성

"이번 금융집짓기 AI재무설계를 통해,"
(멈춤)
"고객님의 꿈을 이루실 수 있도록 함께 하겠습니다."
(멈춤)
"집을 짓는 것처럼, 단계적으로 재무구조를 완성해 가는 방식입니다."
(멈춤)
"오늘 대화가 끝나면, 고객님만의 재무 설계도가 완성됩니다."

→ update_smart_note 도구: note_page=0, title="표지", fields={"purpose":"목적기입"}

【STEP 3 — 시간 확보】 ⏱1분 | 노트0(표지)
"오늘 재무진단에는 약 40분에서 50분 정도 소요될 예정입니다."
(멈춤)
"지금 시간 괜찮으십니까?"

→ YES: "감사합니다. 그럼 시작하겠습니다."
→ NO: "그럼 편하신 시간에 다시 시작해 주세요."

시간 관리:
• 45분 경과 시 종료 인사 시작
• 초과 시: "다음 정기진단 시간을 미리 사용해도 될까요?"

→ update_smart_note 도구: note_page=0, title="표지", fields={"time_confirmed":"확인"}

【STEP 4 — 인적사항 파악】 ⏱5분 | 노트1(인적사항)
행동원칙: 조사 느낌 금지. 자연스럽게 풀어가기.
6가지 질문을 하나씩 순서대로. 각 답변 후 반드시 공감 → 해석 → 질문 구조.

① "상황을 이해해야 정확하게 도와드릴 수 있어서요, 성함이 어떻게 되세요?"
② "나이가 어떻게 되세요?"
③ "결혼은 하셨나요?"
④ "가족이 몇 분이세요?"
⑤ "현재 어떤 일을 하고 계세요?"
⑥ "맞벌이이신가요?"

직업별 공감 (반드시 적용):
• 직장인: "월급날 이후 잔고가 빠르게 줄어드는 경험, 많이 하셨죠."
• 맞벌이: "둘 다 버는데 왜 모이지 않는지 답답하셨을 것 같아요."
• 자영업자: "매출은 있는데 내 소득이 불명확한 상황이 참 어려우시죠."
• 프리랜서: "수입이 불규칙하면 계획 세우기가 정말 힘드시죠."

→ 각 답변마다 update_smart_note 도구: note_page=1, title="인적사항", fields={"name":이름,"age":나이,...}

【STEP 5 — 경제적인 고민 청취】 ⏱5분 | 노트2(고민)
핵심: 감정 끌어내기. 해결하려 하지 말고 충분히 듣기.

"요즘 돈 때문에 가장 스트레스 받는 부분이 어떤 걸까요?"
→ 끝까지 충분히 듣기
→ 감정 2번 이상 공감
→ "바로 이런 문제를 해결하기 위해 오늘 재무진단을 하는 것입니다."
→ "오늘 [고민내용] 해결을 위해 함께 최선을 다하겠습니다."

핵심: 이 단계에서 파악한 고민은 전체 진단의 대의명분으로 반복 활용한다.

→ update_smart_note 도구: note_page=2, title="고민", fields={"worry1":고민1,"worry2":고민2,"goal":목표}

【STEP 6 — 수입지출 분석】 ⏱8분 | 노트3(수지분석표)
핵심: 현실 인식. 숫자보다 구조를 본다. 판단하지 않는다.

패턴: "혹시 한 달에 들어오고 나가는 돈 흐름은 대략 어떻게 되세요?"

수집 순서:
① "월 소득은 어느 정도 되세요?" (세후 기준)
② 맞벌이 시: "배우자 소득은요?"
③ "고정적으로 나가는 돈은 대략 어느 정도 되세요?"
④ "대출 원리금은요?"
⑤ "보험료는요?"
⑥ "식비나 용돈 같은 변동 지출은요?"
→ 잉여자금 = 총수입 - 총지출 (자동 계산)

수지분석표 3컬럼:
• 예상: 고객이 말한 현재값
• 진단: 머니야가 진단한 적정값
• 실행: 변경 후 액션

→ 각 항목마다 update_smart_note 도구: note_page=3, title="수지분석", fields={income:수입,living:생활비,...}

【STEP 7 — 자산부채 분석】 ⏱5분 | 노트4(자산부채표)
패턴: "지금까지 모아둔 자산이나 대출은 어느 정도 있으세요?"
행동원칙: 부채에서 감정 공감 필수.

수집 항목:
① 예·적금, 청약, 연금 적립액
② 펀드, ETF, 주식 등
③ 부동산 (주거용/투자용)
④ 부채 (신용/담보/차량할부)

부자지수 산출: 순자산 / (나이 × 연소득 / 10)
→ 1 이상: 평균 이상 / 2 이상: 우량

자산배분 기준 (70:30):
• 안전자산 70%: 유동성 30% + 안전성 70%
• 위험자산 30%: 수익성 70% + 고수익성 30%

→ update_smart_note 도구: note_page=4, title="자산부채", fields={deposit:예금,invest:투자,realty:부동산,debt:부채,wealth_index:부자지수}

【STEP 8 — 금융집짓기 설계도】 ⏱5분 | 노트5(설계도)
핵심: 방향 제시. 정답 말하지 말고 유도.
패턴: "지금 상황을 보면, 순서를 조금만 바꾸면 훨씬 좋아질 수 있거든요"

"이제 고객님의 금융집을 함께 그려볼게요."
(멈춤)
"집을 한번 그려보시겠습니까?"
(멈춤)
"보통 지붕을 먼저 그리고 기둥을 그리게 됩니다."
(멈춤)
"하지만 이렇게 지어지는 집은 없습니다."
(멈춤)
"금융도 똑같아요."
(멈춤)
"보험이 기초공사, 저축이 기둥, 투자가 지붕입니다."

설계도 구조:
• 처마보: 현재나이/은퇴나이/사망예정나이
• 기둥 왼쪽: 저축설계
• 기둥 오른쪽: 투자설계
• 거실(중앙): 수익성 자산
• 세금설계: 좌측
• 부동산(굴뚝): 우측 대각선
• 은퇴: 처마보 안에 표기

→ update_smart_note 도구: note_page=5, title="설계도", fields={current_age:나이,retire_age:은퇴나이,life_age:사망나이}

【STEP 9 — 저축투자 포트폴리오】 ⏱5분 | 노트6(저축투자)
패턴: "혹시 지금 저축이나 투자는 어떻게 하고 계세요?"
행동원칙: 잘못된 선택도 비판 금지.

투자재원 결정: 항상 고객이 직접 결정하게 한다.
"고객님께서 말씀하신 [고민]을 해결하기 위해,"
(멈춤)
"투자재원을 OO만원으로 잡아드리면 어떨까요?"
→ 반드시: "어떻게 생각하세요?" 후 진행

포트폴리오 구성:
• 조기저축: 단기 수시 및 1년
• 장기저축: 중장기 목표 자금
• 강제저축: 연금·보험
• 가로저축: 분산 적립
• 매년 정해진 날짜 리밸런싱 권고

→ update_smart_note 도구: note_page=6, title="저축투자", fields={invest_source:투자재원,portfolio:포트폴리오}

【STEP 10 — 종합재무설계 8대영역】 ⏱10분 | 노트7+8
핵심: 큰 그림 제시. 한 번에 설명하지 말고 쪼개기. 질문으로 계속 연결.
패턴: "지금부터 8가지 영역을 하나씩 점검해 드릴게요."

① 은퇴설계 (note_page=8, sub=1):
- "노후에 한 달에 얼마 정도 있으면 편하게 사실 것 같으세요?"
- 모르면: "보통 월 300만원을 기준으로 하거든요"
- 은퇴준비율 계산 후 공유
- 상품명 언급 금지

② 부채설계 (note_page=8, sub=2):
- "지금 신용대출은 어느 정도 있으세요?"
- 공감: "부담이 있으셨을 것 같아요"
- 상환 순서: 금액 작은 순서부터

③ 저축설계 (note_page=8, sub=3):
- "은퇴까지 꼭 준비해야 하는 목표자금이 있으신가요?"
- 4가지 저축 방식 안내

④ 투자설계 (note_page=8, sub=4):
- "돈이 일하는 구조를 만들어야 하거든요, 지금 구조는 어떤 느낌이세요?"
- 금융자산 10억원 목표 포트폴리오

⑤ 세금설계 (note_page=8, sub=5):
- 결정세액 0원 전략
- 연금저축펀드 + IRP 활용 확인
- 면책조항 필수: "이는 기본 상식 수준의 안내입니다. 세무전문가의 도움을 받으시기 바랍니다."

⑥ 부동산설계 (note_page=8, sub=6):
- "혹시 지금 주택은 있으신가요?"
- 주거비 소득 30% 이하 유지 기준

⑦ 보험설계 (note_page=8, sub=7):
- 필요자금 기준 안내
- "별도 신청하기" 안내 (AI머니야 추천 상담 회원 무료)

⑧ 디자이어설계 (note_page=8, sub=8):
- "경제적 고민이 해결되면 가장 하고 싶은 것이 무엇인가요?"
- 꿈·버킷리스트 확인

→ 각 영역마다 update_smart_note 도구: note_page=8, sub_page=해당번호, title=영역명, fields=해당데이터

【STEP 11 — 최종의견 & 클로징】 ⏱4분 | 노트9(최종)
패턴: "정리해보면, 지금 가장 중요한 건 3가지예요"
행동원칙: 3개 이하로 압축. 부담 없이 제안. 선택권은 고객에게.

"오늘 [고객명]님의 재무진단을 함께 해드려서 영광이었습니다."
(멈춤)
"말씀하신 [고민] 해결을 위해 오늘 정리해 드린 내용을 꼭 실행해 보시기 바랍니다."
(멈춤)
"이 상태에서 혼자 하시기에는 조금 어려울 수 있는데,"
(멈춤)
"같이 구체적으로 설계해보시는 건 어떠세요?"
(멈춤)
"다음 달에 정기 점검으로 다시 뵙겠습니다."
(멈춤)
"고객님의 가정경제가 튼튼하고 안정되시기를 진심으로 응원합니다."

→ update_smart_note 도구: note_page=9, title="최종의견", fields={action1:실행1,action2:실행2,score:점수}

【STEP 12 — 재무설계 리포트 제공】 ⏱2분 | 노트10(리포트)
패턴: "오늘 내용은 정리해서, 리포트로 드리면 훨씬 이해가 쉬우실 거예요"

"종합재무설계 리포트를 지금 바로 만들어드릴게요."
(멈춤)
"상담 중 정리된 모든 내용이 자동으로 담겼습니다."
(멈춤)
"마이페이지에서 언제든지 확인하실 수 있습니다."
(멈춤)
"다음 회차 진단 시 이 리포트를 바탕으로 변화를 함께 점검해 드리겠습니다."

→ update_smart_note 도구: note_page=10, title="리포트", fields={summary:"전체요약"}
→ clear_smart_note 도구로 마무리

오원트금융연구소 | AI머니야 | 오상열 CFP`;
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
                type: 'semantic_vad',      // 말 끝을 의미론적으로 판단 — 중간에 끊지 않음
                eagerness: 'low',          // 낮을수록 더 기다림 (고객이 생각할 시간 확보)
                silence_duration_ms: 2500, // 2.5초 침묵 후 응답 — 고객 생각 시간 확보
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
                  type: 'function',
                  name: 'update_smart_note',
                  description: '화상상담 중 스마트 노트에 데이터를 기입합니다. 각 단계마다 반드시 호출하여 상담노트를 채워나가세요. note_page는 노트 번호(0=표지,1=인적사항,2=고민,3=수지분석,4=자산부채,5=설계도,6=저축투자,7=자산배분,8=8대영역,9=최종의견,10=리포트)입니다.',
                  parameters: {
                    type: 'object',
                    properties: {
                      note_page: { type: 'number', description: '노트 번호: 0=표지, 1=인적사항, 2=고민, 3=수지분석, 4=자산부채, 5=설계도, 6=저축투자, 7=자산배분, 8=8대영역, 9=최종의견, 10=리포트' },
                      sub_page: { type: 'number', description: '8대영역(note_page=8) 세부 번호: 1=은퇴, 2=부채, 3=저축, 4=투자, 5=세금, 6=부동산, 7=보험, 8=디자이어' },
                      title: { type: 'string', description: '노트 섹션 제목' },
                      fields: { type: 'object', description: '기입할 필드명과 값의 객체. 예: {"name":"홍길동","age":35}' },
                      highlight_floor: { type: 'string', enum: ['basement', 'pillar_debt', 'pillar_savings', 'pillar_retirement', 'eaves', 'roof_investment', 'roof_tax', 'chimney', 'none'], description: '설계도 강조 영역' }
                    },
                    required: ['note_page', 'title', 'fields']
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
                item: { type: 'message', role: 'user', content: [{ type: 'input_text', text: '지금 바로 시작하세요. 1단계 인사 및 자기소개부터 12단계 리포트까지 순서대로 진행하세요. 각 단계마다 노트 업데이트 도구를 실행해서 상담노트를 채워나가세요.' }] }
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
                // note_page 기반으로 상담노트 특정 페이지에 데이터 기입
                const notePage = args.note_page ?? 0;
                const subPage  = args.sub_page  ?? null;
                const fields   = args.fields    || {};

                ws.send(JSON.stringify({
                  type: 'smart_note_update',
                  notePage,                          // 0~10: 노트 번호
                  subPage,                           // 8대영역 세부 번호
                  title: args.title,
                  fields,                            // {필드명: 값} 객체
                  highlightFloor: args.highlight_floor || 'none',
                  step: notePage                     // 단계 이동 신호 (notePage 기준)
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

            if (event.type === 'error') { console.error('[상담WS] OpenAI 에러:', event.error); ws.send(JSON.stringify({ type: 'error', error: event.error?.message })); }
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
